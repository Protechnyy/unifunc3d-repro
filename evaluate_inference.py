#!/usr/bin/env python3
"""Evaluate NEW predictions and released 8B masks on the identical query subset."""
import argparse
from pathlib import Path

import numpy as np

from release_lib.common import (ROOT, METRICS, Scene, read_json, scores_for_indices,
                                validate_indices, write_json)


def evaluate(dataset_root, split, prediction_dir, metadata_overrides=None):
    rows = []
    from unifunc3d.metadata import benchmark_metadata
    metas, exceptions = benchmark_metadata(split, metadata_overrides)
    for path in sorted(Path(prediction_dir).glob("*.json")):
        visit = path.stem
        if visit not in metas:
            raise ValueError(f"Unknown scene {visit}")
        scene = Scene(dataset_root, split, visit)
        if scene.metadata() != metas[visit]:
            raise ValueError(f"Dataset fingerprint mismatch: {visit}")
        released = {p["desc_id"]: p["pred_mask"] for p in read_json(ROOT / "predictions" / f"8b_{split}" / path.name)}
        seen = set()
        for prediction in read_json(path):
            desc_id = prediction["desc_id"]
            if desc_id in seen or desc_id not in scene.by_id:
                raise ValueError(f"Duplicate/unknown query {visit}/{desc_id}")
            seen.add(desc_id)
            indices = validate_indices(prediction["pred_mask"], scene.n_points)
            gt = scene.gt_indices(desc_id)
            rows.append({"visit_id": visit, "desc_id": desc_id,
                         "description": scene.by_id[desc_id]["description"],
                         "ours": scores_for_indices(gt, indices),
                         "released_8b": scores_for_indices(gt, np.array(released[desc_id], dtype=np.int64))})
    if not rows:
        raise ValueError("No predictions to evaluate")
    means = {label: {m: float(np.mean([r[label][m] for r in rows])) for m in METRICS}
             for label in ("ours", "released_8b")}
    coverage = {visit: {"evaluated": sum(r["visit_id"] == visit for r in rows),
                        "total": len(metas[visit]["desc_ids"])}
                for visit in sorted({r["visit_id"] for r in rows})}
    return {"split": split, "metadata_overrides": exceptions,
            "release_fingerprints_exact": not bool(exceptions),
            "queries": len(rows), "scenes": len({r["visit_id"] for r in rows}),
            "complete_split": len(rows) == sum(len(m["desc_ids"]) for m in metas.values()),
            "coverage": coverage,
            "complete_selected_scenes": all(c["evaluated"] == c["total"] for c in coverage.values()),
            "empty_predictions": sum(r["ours"]["pred_count"] == 0 for r in rows),
            "means": means, "percent": {k: {m: round(v * 100, 4) for m, v in ms.items()} for k, ms in means.items()},
            "protocol": "Historical Fun3DU point precision/recall thresholds; NOT instance AP",
            "per_query": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scenefun3d"))
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-overrides", type=Path)
    args = parser.parse_args()
    result = evaluate(args.dataset_root, args.split, args.predictions, args.metadata_overrides)
    write_json(args.output, result, pretty=True)
    print(f"{result['scenes']} scenes, {result['queries']} queries, complete_split={result['complete_split']}")
    for label, values in result["percent"].items():
        print(label, values)


if __name__ == "__main__":
    main()
