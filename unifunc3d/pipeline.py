"""Resumable inference. Ground truth is used only by the separate evaluator."""
import hashlib
import fcntl
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from release_lib.common import read_json, sha256, write_json
from .data import InputScene
from .geometry import agreement, lift_frame
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

    def close(self):
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
        for (frame_key, obj), (frame, _) in candidates.items():
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
                image = frame.image(config.upright)
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
                indices = lift_frame(scene.xyz, verified, frame, config.upright,
                                     config.visibility_tolerance) if verified.any() else np.empty(0, dtype=np.int64)
                record = {"frame": frame_key, "object": obj, "points": parsed["points"],
                          "masks": mask_records, "lifted_points": len(indices)}
                # NPZ first; the JSON file is the commit marker for a completed frame.
                with mask_path.with_suffix(".npz.tmp").open("wb") as f:
                    np.savez_compressed(f, indices=indices)
                mask_path.with_suffix(".npz.tmp").replace(mask_path)
                write_json(record_path, record, pretty=True)
            accepted_per_frame.setdefault(frame_key, []).append(indices)
            fine_records.append(record)
        for chunks in accepted_per_frame.values():
            indices = np.unique(np.concatenate(chunks))
            votes[indices] += 1
        prediction = agreement(votes, config.agreement_threshold)
        summary = {"visit_id": scene.visit_id, "desc_id": description["desc_id"],
                   "coarse_calls": len(coarse_records), "fine_candidates": len(candidates),
                   "frames_with_accepted_points": sum(any(len(x) for x in xs) for xs in accepted_per_frame.values()),
                   "max_votes": int(votes.max(initial=0)), "predicted_points": len(prediction),
                   "seconds_this_invocation": time.monotonic() - started,
                   "fine": fine_records}
        write_json(directory / "summary.json", summary, pretty=True)
        nz = np.flatnonzero(votes)
        np.savez_compressed(directory / "votes.npz", indices=nz, votes=votes[nz], n_points=len(votes))
        return {"desc_id": description["desc_id"], "pred_mask": prediction.tolist()}

    def run(self, root, split, visits, limit_queries=0, desc_ids=None):
        pred_dir = self.output / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for visit in visits:
            self.coarse_images.clear()
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
