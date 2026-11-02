"""Resumable inference. Ground truth is used only by the separate evaluator."""
import hashlib
import fcntl
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from release_lib.common import read_json, sha256, write_json
from .data import InputScene
from .geometry import agreement, ProjectionCache
from .grounding import (localization_prompt, parse_grounding, parse_selection, parse_verification,
                        sample_indices, selection_prompt, verify_prompt)
from .models import Qwen, SAM3, overlay
from .image_cache import ImageCache

log = logging.getLogger(__name__)
HYPOTHESIS_FIELDS = ("target_object", "functional_element", "affordance_type", "instance_constraints")


def hypothesis_suffix(record):
    return hashlib.sha256(json.dumps({k: record[k] for k in HYPOTHESIS_FIELDS},
                                     sort_keys=True).encode()).hexdigest()[:12]


class Pipeline:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.qwen = None
        self.sam = None
        self.output.mkdir(parents=True, exist_ok=True)
        self._lock = (self.output / ".run.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise RuntimeError(f"Another inference process is using {self.output}") from exc
        config_path = self.output / "config.json"
        recorded = asdict(config)
        if config_path.exists() and read_json(config_path) != recorded:
            self._lock.close()
            raise ValueError("Output has a different configuration; choose a new output directory")
        write_json(config_path, recorded, pretty=True)
        # Original-resolution frames, reused across descriptions of the same scene.
        self.images = ImageCache()
        self.projection_pool = ThreadPoolExecutor(max_workers=1)
        self.projection_cache = None

    def close(self):
        self.projection_pool.shutdown(wait=True)
        self.images.close()
        self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def ask(self, path, images, prompt, **kwargs):
        if path.exists():
            return read_json(path)
        if self.qwen is None:
            self.qwen = Qwen(self.config)
        result = self.qwen.ask(images, prompt, **kwargs)
        result["prompt"] = prompt
        write_json(path, result, pretty=True)
        return result

    def select(self, directory, scene, task):
        config = self.config
        records = []
        candidates = {}
        for video_id, frames in scene.videos.items():
            for iteration in range(config.iterations):
                sampled = sample_indices(len(frames), config.frames_per_round, config.iterations, iteration)
                path = directory / f"select_{video_id}_{iteration}.json"
                images = []
                if not path.exists():
                    images = self.images.get([frames[i] for i in sampled], config.upright)
                response = self.ask(path, images, selection_prompt(task),
                                    labels=[f"<frame {i}>:" for i in range(len(sampled))])
                parsed = parse_selection(response["text"], len(sampled), config.max_selected_frames)
                records.append({"video_id": video_id, "iteration": iteration, "sampled_indices": sampled,
                                **parsed, "frames": [frames[sampled[i]].key for i in parsed["frame_ids"]]})
                if parsed["error"]:
                    log.warning("Invalid selection %s: %s", path, parsed["error"])
                    continue
                hypothesis = {k: parsed[k] for k in HYPOTHESIS_FIELDS}
                for i in parsed["frame_ids"]:
                    frame = frames[sampled[i]]
                    # Localize each (frame, hypothesis) pair once.
                    candidates.setdefault((frame.key, json.dumps(hypothesis, sort_keys=True)),
                                          {"frame": frame, "hypothesis": hypothesis})
        return records, list(candidates.values())

    def localize(self, directory, task, candidate, prefix):
        config = self.config
        frame, hypothesis = candidate["frame"], candidate["hypothesis"]
        image = self.images.get([frame], config.upright)[0]
        response = self.ask(directory / f"ground_{prefix}.json", [image],
                            localization_prompt(task, hypothesis))
        points = parse_grounding(response["text"])
        verified = np.zeros((image.height, image.width), dtype=bool)
        mask_records = []
        if points:
            if self.sam is None:
                self.sam = SAM3(config)
            for m, candidate_mask in enumerate(self.sam.segment(image, points)):
                highlighted = overlay(image, candidate_mask["mask"], config.overlay_alpha)
                verification = self.ask(directory / f"verify_{prefix}_{m}.json", [highlighted],
                                        verify_prompt(task, hypothesis))
                accepted = parse_verification(verification["text"])
                highlighted.thumbnail((960, 960))
                highlighted.save(directory / f"overlay_{prefix}_{m}.jpg", quality=90)
                mask_records.append({"point": candidate_mask["point"], "score": candidate_mask["score"],
                                     "pixels": int(candidate_mask["mask"].sum()), "accepted": accepted,
                                     "verification": verification["text"]})
                if accepted:
                    verified |= candidate_mask["mask"]
        record = {"frame": frame.key, **hypothesis, "response": response["text"],
                  "points": points, "masks": mask_records}
        return record, verified

    def project_and_save(self, record, verified, frame, mask_path, record_path):
        config = self.config
        indices = self.projection_cache.lift(verified, frame, config.upright,
                                             config.visibility_tolerance) if verified.any() else np.empty(0, dtype=np.int64)
        record["lifted_points"] = len(indices)
        with mask_path.with_suffix(".npz.tmp").open("wb") as f:
            np.savez_compressed(f, indices=indices)
        mask_path.with_suffix(".npz.tmp").replace(mask_path)
        write_json(record_path, record, pretty=True)
        return record, indices

    def query(self, scene, description):
        config = self.config
        started = time.monotonic()
        directory = self.output / "traces" / scene.visit_id / description["desc_id"]
        directory.mkdir(parents=True, exist_ok=True)
        desc_path = directory / "description.json"
        if desc_path.exists() and read_json(desc_path) != description:
            raise ValueError("Description changed since cache creation")
        write_json(desc_path, description, pretty=True)
        task = description["description"]
        selection_records, candidates = self.select(directory, scene, task)
        write_json(directory / "selection_summary.json", {
            "calls": selection_records,
            "candidates": [{"frame": c["frame"].key, **c["hypothesis"]} for c in candidates]}, pretty=True)
        log.info("%s: %d frame/hypothesis candidates", description["desc_id"], len(candidates))
        if self.projection_cache is None or self.projection_cache.xyz is not scene.xyz:
            self.projection_cache = ProjectionCache(scene.xyz)
        # One vote per camera frame; hypotheses on the same frame are merged first.
        accepted_per_frame = {}
        localization_records = []

        def collect(frame_key, record, indices):
            accepted_per_frame.setdefault(frame_key, []).append(indices)
            localization_records.append(record)

        pending = None
        for candidate in candidates:
            frame = candidate["frame"]
            # Frame keys contain decimal timestamps; with_suffix() would alias frames.
            prefix = f"{frame.key}_{hypothesis_suffix(candidate['hypothesis'])}"
            record_path = directory / f"localize_{prefix}.json"
            mask_path = directory / f"localize_{prefix}.npz"
            if record_path.exists() and mask_path.exists():
                with np.load(mask_path, allow_pickle=False) as saved:
                    collect(frame.key, read_json(record_path), saved["indices"])
                continue
            record, verified = self.localize(directory, task, candidate, prefix)
            # Complete the previous projection before queuing another, so CPU
            # work and retained masks stay bounded while overlapping GPU work.
            if pending is not None:
                collect(pending[0], *pending[1].result())
            pending = (frame.key, self.projection_pool.submit(
                self.project_and_save, record, verified, frame, mask_path, record_path))
        if pending is not None:
            collect(pending[0], *pending[1].result())
        votes = np.zeros(len(scene.xyz), dtype=np.uint32)
        for chunks in accepted_per_frame.values():
            votes[np.unique(np.concatenate(chunks))] += 1
        prediction = agreement(votes, config.agreement_threshold)
        summary = {"visit_id": scene.visit_id, "desc_id": description["desc_id"],
                   "selection_calls": len(selection_records),
                   "invalid_selection_calls": sum(r["error"] is not None for r in selection_records),
                   "candidates": len(candidates),
                   "frames_with_accepted_points": sum(any(len(x) for x in xs) for xs in accepted_per_frame.values()),
                   "max_votes": int(votes.max(initial=0)), "predicted_points": len(prediction),
                   "localization": localization_records}
        nz = np.flatnonzero(votes)
        np.savez_compressed(directory / "votes.npz", indices=nz, votes=votes[nz], n_points=len(votes))
        summary["seconds_this_invocation"] = time.monotonic() - started
        summary["projection_cache"] = {"hits": self.projection_cache.hits, "misses": self.projection_cache.misses,
                                       "bytes": self.projection_cache.bytes}
        write_json(directory / "summary.json", summary, pretty=True)
        return {"desc_id": description["desc_id"], "pred_mask": prediction.tolist()}

    def run(self, root, split, visits, limit_queries=0, desc_ids=None):
        pred_dir = self.output / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for visit in visits:
            self.images.clear()
            self.projection_cache = None
            scene = InputScene(root, split, visit, self.config.max_videos)
            snapshot = {"dataset_root": str(Path(root).resolve()), "split": split,
                        "n_points": len(scene.xyz), "files": {}}
            for suffix in ("descriptions.json", "crop_mask.npy"):
                path = scene.directory / f"{visit}_{suffix}"
                snapshot["files"][path.name] = sha256(path)
            scan = scene.directory / f"{visit}_laser_scan.ply"
            snapshot["files"][scan.name] = {"bytes": scan.stat().st_size, "mtime_ns": scan.stat().st_mtime_ns}
            for video, frames in scene.videos.items():
                snapshot["files"][video] = {
                    "trajectory_sha256": sha256(scene.directory / video / "hires_poses.traj"),
                    "frames": [[f.key, f.rgb.stat().st_size, f.rgb.stat().st_mtime_ns,
                                f.depth.stat().st_size, f.depth.stat().st_mtime_ns,
                                f.intrinsics.stat().st_size, f.intrinsics.stat().st_mtime_ns] for f in frames]}
            snapshot_path = self.output / "inputs" / f"{visit}.json"
            if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
                raise ValueError(f"Input assets changed for {visit}; choose a new output directory")
            write_json(snapshot_path, snapshot)
            destination = pred_dir / f"{visit}.json"
            existing = {p["desc_id"]: p for p in read_json(destination)} if destination.exists() else {}
            descriptions = scene.descriptions
            if desc_ids:
                descriptions = [d for d in descriptions if d["desc_id"] in desc_ids]
            if limit_queries:
                descriptions = descriptions[:limit_queries]
            for i, description in enumerate(descriptions):
                if description["desc_id"] in existing:
                    continue
                log.info("Scene %s query %d/%d: %s", visit, i + 1, len(descriptions), description["description"])
                result = self.query(scene, description)
                existing[result["desc_id"]] = result
                write_json(destination, list(existing.values()))
            log.info("Scene %s: %d saved predictions", visit, len(existing))
