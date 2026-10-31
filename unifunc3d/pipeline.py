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
from .grounding import (coarse_indices, coarse_prompt, fine_prompt, parse_grounding,
                        parse_verification, verify_prompt, window_indices)
from .models import Qwen, SAM3, overlay
from .image_cache import CoarseImageCache

log = logging.getLogger(__name__)


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
        self.coarse_images = CoarseImageCache()
        self.prefetch_pool = ThreadPoolExecutor(max_workers=2)
        self.projection_pool = ThreadPoolExecutor(max_workers=1)
        self.projection_cache = None

    def close(self):
        self.prefetch_pool.shutdown(wait=True)
        self.projection_pool.shutdown(wait=True)
        self.coarse_images.close()
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
        candidates = {}
        coarse_records = []
        for video_id, frames in scene.videos.items():
            for iteration in range(config.iterations):
                sampled = coarse_indices(len(frames), config.coarse_frames, config.iterations, iteration)
                path = directory / f"coarse_{video_id}_{iteration}.json"
                images = []
                if not path.exists():
                    images = self.coarse_images.get([frames[i] for i in sampled], config.upright,
                                                    config.coarse_width, config.coarse_height)
                response = self.ask(path, images, coarse_prompt(task, config.part_aware_prompt), tagged=True,
                                    max_pixels=config.coarse_width * config.coarse_height)
                parsed = parse_grounding(response["text"], len(sampled))
                record = {"video_id": video_id, "iteration": iteration,
                          "sampled_indices": sampled, **parsed}
                coarse_records.append(record)
                if parsed["frame"] is None or not parsed["functional_object"]:
                    log.warning("Invalid coarse candidate: %s", path)
                    continue
                center = sampled[parsed["frame"]]
                selected = window_indices([f.timestamp for f in frames], center, config.coarse_frames)
                if config.max_fine_frames and len(selected) > config.max_fine_frames:
                    subset = np.linspace(0, len(selected) - 1, config.max_fine_frames).round().astype(int)
                    selected = [selected[i] for i in subset]
                for i in selected:
                    # Retain different semantic hypotheses, deduplicate identical frame/object pairs.
                    obj = parsed["functional_object"]
                    candidates[(frames[i].key, obj)] = (frames[i], obj)
        write_json(directory / "coarse_summary.json", coarse_records, pretty=True)
        log.info("%s: %d fine frame/object candidates", description["desc_id"], len(candidates))
        votes = np.zeros(len(scene.xyz), dtype=np.uint32)
        # One vote per unique camera frame; overlapping windows must not multiply votes.
        accepted_per_frame = {}
        fine_records = []
        if self.projection_cache is None or self.projection_cache.xyz is not scene.xyz:
            self.projection_cache = ProjectionCache(scene.xyz)
        jobs = []
        for (frame_key, obj), (frame, _) in candidates.items():
            suffix = hashlib.sha256(obj.encode()).hexdigest()[:12]
            prefix = directory / f"fine_{frame_key}_{suffix}"
            cached = (directory / (prefix.name + ".json")).exists() and (directory / (prefix.name + ".npz")).exists()
            jobs.append((frame_key, obj, frame, cached))
        image_futures = {}
        # Bound decoded images to the current frame plus two lookahead frames.
        def prefetch(index):
            if index < len(jobs) and not jobs[index][3]:
                image_futures[index] = self.prefetch_pool.submit(jobs[index][2].image, config.upright)
        for index in range(min(3, len(jobs))):
            prefetch(index)
        pending_projection = None
        def collect_projection(pending):
            record, indices = pending.result()
            accepted_per_frame.setdefault(record['frame'], []).append(indices)
            fine_records.append(record)
        def project_and_save(verified, frame, parsed, mask_records, obj, mask_path, record_path):
            indices = self.projection_cache.lift(verified, frame, config.upright,
                                                config.visibility_tolerance) if verified.any() else np.empty(0, dtype=np.int64)
            record = {"frame": frame.key, "object": obj, "points": parsed["points"],
                      "masks": mask_records, "lifted_points": len(indices)}
            with mask_path.with_suffix(".npz.tmp").open("wb") as f:
                np.savez_compressed(f, indices=indices)
            mask_path.with_suffix(".npz.tmp").replace(mask_path)
            write_json(record_path, record, pretty=True)
            return record, indices
        for job_index, (frame_key, obj, frame, cached) in enumerate(jobs):
            suffix = hashlib.sha256(obj.encode()).hexdigest()[:12]
            prefix = directory / f"fine_{frame_key}_{suffix}"
            # Frame keys contain decimal timestamps. with_suffix() would remove
            # the fractional timestamp AND the object hash, aliasing frames.
            record_path = prefix.parent / (prefix.name + ".json")
            mask_path = prefix.parent / (prefix.name + ".npz")
            if record_path.exists() and mask_path.exists():
                record = read_json(record_path)
                with np.load(mask_path, allow_pickle=False) as saved:
                    indices = saved["indices"]
            else:
                image = image_futures.pop(job_index).result()
                prefetch(job_index + 3)
                response = self.ask(directory / f"ground_{frame_key}_{suffix}.json", [image], fine_prompt(task, obj))
                parsed = parse_grounding(response["text"])
                verified = np.zeros((image.height, image.width), dtype=bool)
                mask_records = []
                if parsed["points"]:
                    if self.sam is None:
                        self.sam = SAM3(config)
                    masks = self.sam.segment(image, parsed["points"])
                    for m, candidate in enumerate(masks):
                        highlighted = overlay(image, candidate["mask"], config.overlay_alpha)
                        verification = self.ask(directory / f"verify_{frame_key}_{suffix}_{m}.json",
                                                [highlighted], verify_prompt(obj))
                        accepted = parse_verification(verification["text"])
                        highlighted.thumbnail((960, 960))
                        highlighted.save(directory / f"overlay_{frame_key}_{suffix}_{m}.jpg", quality=90)
                        mask_records.append({"point": candidate["point"], "score": candidate["score"],
                                             "pixels": int(candidate["mask"].sum()), "accepted": accepted,
                                             "verification": verification["text"]})
                        if accepted:
                            verified |= candidate["mask"]
                # Complete the previous projection before queuing another, so CPU
                # work and retained masks stay bounded while overlapping GPU work.
                if pending_projection is not None:
                    collect_projection(pending_projection)
                pending_projection = self.projection_pool.submit(
                    project_and_save, verified, frame, parsed, mask_records, obj, mask_path, record_path)
                continue
            prefetch(job_index + 3)
            accepted_per_frame.setdefault(frame_key, []).append(indices)
            fine_records.append(record)
        if pending_projection is not None:
            collect_projection(pending_projection)
        for chunks in accepted_per_frame.values():
            indices = np.unique(np.concatenate(chunks))
            votes[indices] += 1
        prediction = agreement(votes, config.agreement_threshold)
        summary = {"visit_id": scene.visit_id, "desc_id": description["desc_id"],
                   "coarse_calls": len(coarse_records), "fine_candidates": len(candidates),
                   "frames_with_accepted_points": sum(any(len(x) for x in xs) for xs in accepted_per_frame.values()),
                   "max_votes": int(votes.max(initial=0)), "predicted_points": len(prediction),
                   "seconds_this_invocation": time.monotonic() - started,
                   "projection_cache": {"hits": self.projection_cache.hits,
                                        "misses": self.projection_cache.misses,
                                        "bytes": self.projection_cache.bytes},
                   "fine": fine_records}
        if self.qwen is not None and hasattr(self.qwen, "visual_cache"):
            summary["visual_cache"] = {"hits": self.qwen.visual_cache.hits,
                                       "misses": self.qwen.visual_cache.misses,
                                       "bytes": self.qwen.visual_cache.bytes}
        write_json(directory / "summary.json", summary, pretty=True)
        nz = np.flatnonzero(votes)
        np.savez_compressed(directory / "votes.npz", indices=nz, votes=votes[nz], n_points=len(votes))
        return {"desc_id": description["desc_id"], "pred_mask": prediction.tolist()}

    def run(self, root, split, visits, limit_queries=0, desc_ids=None):
        pred_dir = self.output / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for visit in visits:
            self.coarse_images.clear()
            self.projection_cache = None
            if self.qwen is not None:
                self.qwen.visual_cache.clear()
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
