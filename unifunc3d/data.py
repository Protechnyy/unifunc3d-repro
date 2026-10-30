"""SceneFun3D RGB-D loader. Inference never reads annotation masks or released predictions."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from release_lib.common import ROOT, sha256


@dataclass
class Frame:
    video_id: str
    timestamp: float
    rgb: Path
    depth: Path
    intrinsics: Path
    world_to_camera: np.ndarray
    rotation: int

    @property
    def key(self):
        return self.rgb.stem

    def image(self, upright=True):
        with Image.open(self.rgb) as im:
            arr = np.asarray(im.convert("RGB"))
        return Image.fromarray(np.rot90(arr, -self.rotation).copy()) if upright else Image.fromarray(arr)


def video_ids(root, split):
    result = {}
    with (Path(root) / "benchmark_file_lists" / f"{split}_set.csv").open() as f:
        for row in csv.DictReader(f):
            result.setdefault(row["visit_id"], []).append(row["video_id"])
    return {k: list(dict.fromkeys(v)) for k, v in sorted(result.items())}


def load_video(directory, video_id):
    directory = Path(directory) / video_id
    trajectory = np.loadtxt(directory / "hires_poses.traj", ndmin=2)
    if trajectory.shape[1] != 7 or not np.isfinite(trajectory).all():
        raise ValueError(f"Invalid trajectory: {directory}")
    trajectory = trajectory[np.argsort(trajectory[:, 0])]
    matrices = Rotation.from_rotvec(trajectory[:, 1:4]).as_matrix()
    frames, missing = [], []
    rgb_paths = sorted((directory / "hires_wide").glob("*.jpg"),
                       key=lambda p: float(p.stem.rsplit("_", 1)[1]))
    if not rgb_paths:
        raise ValueError(f"No RGB frames: {directory}")
    for rgb in rgb_paths:
        timestamp = float(rgb.stem.rsplit("_", 1)[1])
        depth = directory / "hires_depth" / (rgb.stem + ".png")
        intrinsics = directory / "hires_wide_intrinsics" / (rgb.stem + ".pincam")
        if not depth.is_file() or not intrinsics.is_file() or min(depth.stat().st_size, intrinsics.stat().st_size) == 0:
            missing.append(rgb.stem)
            continue
        # Official parser skips RGB outside the trajectory time range.
        if not trajectory[0, 0] <= timestamp <= trajectory[-1, 0]:
            continue
        i = int(np.abs(trajectory[:, 0] - timestamp).argmin())
        w2c = np.eye(4)
        w2c[:3, :3] = matrices[i]
        w2c[:3, 3] = trajectory[i, 4:7]
        # Camera-to-world row z expresses gravity in camera coordinates.
        z = matrices[i].T[2]
        rotation = int(np.argmax(np.array([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]]) @ z))
        frames.append(Frame(video_id, timestamp, rgb, depth, intrinsics, w2c, rotation))
    if missing:
        raise ValueError(f"{video_id}: {len(missing)} RGB frames lack depth/intrinsics (first: {missing[0]})")
    if not frames:
        raise ValueError(f"No posed RGB-D frames: {directory}")
    return frames


def audit(root, split="val", decode=False, visits=None, metadata_overrides=None):
    """Asset availability, crop/PLY alignment, and optionally full image decoding."""
    from plyfile import PlyData
    results = []
    from .metadata import benchmark_metadata
    reference, exceptions = benchmark_metadata(split, metadata_overrides)
    benchmark = video_ids(root, split)
    if visits and set(visits) - set(benchmark):
        raise ValueError(f"Unknown visits: {set(visits) - set(benchmark)}")
    for visit, videos in benchmark.items():
        if visits and visit not in visits:
            continue
        directory = Path(root) / split / visit
        entry = {"visit_id": visit, "complete": False, "videos": {}, "errors": []}
        try:
            crop = np.load(directory / f"{visit}_crop_mask.npy", allow_pickle=False)
            cloud = PlyData.read(directory / f"{visit}_laser_scan.ply", mmap="r")
            if crop.ndim != 1 or len(crop) != cloud["vertex"].count or not np.isin(crop, [0, 1]).all():
                raise ValueError("Invalid crop mask or PLY/crop length mismatch")
            entry["n_points"] = int(np.count_nonzero(crop == 1))
            entry["fingerprint_mismatches"] = []
            if visit in reference:
                for suffix, key in (("crop_mask.npy", "crop_sha256"),
                                    ("descriptions.json", "descriptions_sha256"),
                                    ("annotations.json", "annotations_sha256")):
                    if sha256(directory / f"{visit}_{suffix}") != reference[visit][key]:
                        entry["fingerprint_mismatches"].append(key)
            for suffix in ("descriptions", "annotations"):
                value = json.loads((directory / f"{visit}_{suffix}.json").read_text())
                entry[f"n_{suffix}"] = len(value[suffix])
            for video in videos:
                try:
                    frames = load_video(directory, video)
                    if decode:
                        for frame in frames:
                            with Image.open(frame.rgb) as im:
                                im.load()
                                size = im.size
                            with Image.open(frame.depth) as im:
                                im.load()
                                if im.size != size:
                                    raise ValueError(f"RGB/depth size mismatch: {frame.key}")
                            intr = np.loadtxt(frame.intrinsics)
                            if tuple(intr[:2]) != size or not np.isfinite(intr).all():
                                raise ValueError(f"Invalid intrinsics: {frame.key}")
                    entry["videos"][video] = len(frames)
                except (OSError, ValueError) as e:
                    entry["errors"].append(str(e))
            entry["complete"] = not entry["errors"]
        except (OSError, ValueError, KeyError) as e:
            entry["errors"].append(str(e))
        results.append(entry)
    return {"split": split, "decode_checked": decode, "metadata_overrides": exceptions, "scenes": results,
            "complete_scenes": [x["visit_id"] for x in results if x["complete"]],
            "benchmark_compatible_scenes": [x["visit_id"] for x in results
                                             if x["complete"] and not x["fingerprint_mismatches"]]}


class InputScene:
    def __init__(self, root, split, visit_id, max_videos=0):
        from plyfile import PlyData
        self.visit_id = visit_id
        self.directory = Path(root) / split / visit_id
        desc = json.loads((self.directory / f"{visit_id}_descriptions.json").read_text())["descriptions"]
        # Deliberately discard annotation IDs at the inference boundary.
        self.descriptions = [{"desc_id": d["desc_id"], "description": d["description"]} for d in desc]
        videos = video_ids(root, split)[visit_id]
        self.videos = {v: load_video(self.directory, v) for v in videos[:max_videos or None]}
        crop = np.load(self.directory / f"{visit_id}_crop_mask.npy", allow_pickle=False)
        vertices = PlyData.read(self.directory / f"{visit_id}_laser_scan.ply", mmap="r")["vertex"]
        if len(crop) != vertices.count or not np.isin(crop, [0, 1]).all():
            raise ValueError("PLY/crop mismatch")
        # Keep original vertex order. No voxelization or nearest-neighbor reindexing.
        self.xyz = np.column_stack([vertices[c][crop == 1] for c in "xyz"]).astype(np.float32)
