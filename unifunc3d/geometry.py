"""Depth-tested laser point projection and Fun3DU min/max vote normalization."""
import numpy as np
from PIL import Image


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
