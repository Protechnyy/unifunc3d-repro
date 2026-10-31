"""Depth-tested laser point projection and Fun3DU min/max vote normalization."""
import numpy as np
from PIL import Image
from collections import OrderedDict


def project_mask(xyz, mask, depth, intrinsic, world_to_camera, tolerance=.25, chunk_size=250000):
    if mask.shape != depth.shape:
        raise ValueError("Mask/depth must use the same original camera grid")
    h, w = depth.shape
    result = []
    for start in range(0, len(xyz), chunk_size):
        camera = xyz[start:start + chunk_size] @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = camera[:, 2]
        front = np.flatnonzero(np.isfinite(camera).all(axis=1) & (z > 0))
        uv = camera[front, :2] / z[front, None]
        uv = np.rint(uv * [intrinsic[0, 0], intrinsic[1, 1]] + intrinsic[:2, 2]).astype(np.int64)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids, uv = front[inside], uv[inside]
        x, y = uv.T
        d = depth[y, x]
        valid = mask[y, x] & np.isfinite(d) & (d > 0) & (np.abs(d - z[ids]) <= tolerance * d)
        result.append(ids[valid] + start)
    return np.concatenate(result) if result else np.empty(0, dtype=np.int64)


def lift_frame(xyz, upright_mask, frame, upright=True, tolerance=.25):
    mask = np.rot90(upright_mask, frame.rotation).copy() if upright else upright_mask
    with Image.open(frame.depth) as im:
        depth = np.asarray(im, dtype=np.float32) / 1000
    w, h, fx, fy, cx, cy = np.loadtxt(frame.intrinsics)
    if (int(h), int(w)) != depth.shape:
        raise ValueError(f"Intrinsic/depth grid mismatch: {frame.key}")
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return project_mask(xyz, mask.astype(bool), depth, intrinsic, frame.world_to_camera, tolerance)


def agreement(votes, threshold=.7):
    votes = np.asarray(votes, dtype=np.float32)
    if not votes.size or votes.max() == votes.min():
        return np.empty(0, dtype=np.int64)
    normalized = (votes - votes.min()) / (votes.max() - votes.min())
    return np.flatnonzero(normalized > threshold)


class ProjectionCache:
    """Scene-local depth visibility maps, independent of each query's mask.

    Use the same NumPy dtypes, operation order, rounding and depth comparison
    as project_mask. Only the final mask lookup varies between queries.
    """
    def __init__(self, xyz, max_bytes=256 * 1024**2):
        self.xyz = xyz
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def lift(self, upright_mask, frame, upright=True, tolerance=.25):
        mask = np.rot90(upright_mask, frame.rotation).copy() if upright else upright_mask
        key = (str(frame.depth), str(frame.intrinsics), frame.world_to_camera.tobytes(), tolerance)
        if key in self.entries:
            self.hits += 1
            entry = self.entries.pop(key)
            self.entries[key] = entry
        else:
            self.misses += 1
            with Image.open(frame.depth) as im:
                depth = np.asarray(im, dtype=np.float32) / 1000
            w, h, fx, fy, cx, cy = np.loadtxt(frame.intrinsics)
            if (int(h), int(w)) != depth.shape:
                raise ValueError(f"Intrinsic/depth grid mismatch: {frame.key}")
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            h, w = depth.shape
            all_ids, all_pixels = [], []
            for start in range(0, len(self.xyz), 250000):
                camera = self.xyz[start:start + 250000] @ frame.world_to_camera[:3, :3].T + frame.world_to_camera[:3, 3]
                z = camera[:, 2]
                front = np.flatnonzero(np.isfinite(camera).all(axis=1) & (z > 0))
                uv = camera[front, :2] / z[front, None]
                uv = np.rint(uv * [intrinsic[0, 0], intrinsic[1, 1]] + intrinsic[:2, 2]).astype(np.int64)
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                ids, uv = front[inside], uv[inside]
                x, y = uv.T
                d = depth[y, x]
                valid = np.isfinite(d) & (d > 0) & (np.abs(d - z[ids]) <= tolerance * d)
                all_ids.append(ids[valid] + start)
                all_pixels.append((y[valid] * w + x[valid]))
            ids = np.concatenate(all_ids) if all_ids else np.empty(0, dtype=np.int64)
            pixels = np.concatenate(all_pixels) if all_pixels else np.empty(0, dtype=np.int64)
            entry = (ids, pixels, depth.shape)
            size = ids.nbytes + pixels.nbytes
            if size <= self.max_bytes:
                while self.entries and self.bytes + size > self.max_bytes:
                    _, (old_ids, old_pixels, _) = self.entries.popitem(last=False)
                    self.bytes -= old_ids.nbytes + old_pixels.nbytes
                self.entries[key] = entry
                self.bytes += size
        ids, pixels, shape = entry
        if mask.shape != shape:
            raise ValueError("Mask/depth must use the same original camera grid")
        return ids[mask.ravel()[pixels]]
