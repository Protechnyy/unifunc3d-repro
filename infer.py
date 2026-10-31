#!/usr/bin/env python3
"""Audit assets or run the independent UniFunc3D-8B reconstruction."""
import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from release_lib.common import read_json, write_json
from unifunc3d.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "run"])
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scenefun3d"))
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--output", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--visits", nargs="+")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--desc-ids", nargs="+")
    parser.add_argument("--decode", action="store_true", help="Decode all images during asset audit")
    parser.add_argument("--config", type=Path, help="JSON Config overrides")
    parser.add_argument("--metadata-overrides", type=Path, help="Explicit audited description hash exceptions")
    parser.add_argument("--require-full-split", action="store_true", help="Fail rather than silently run a subset")
    args = parser.parse_args()
    if args.limit_scenes < 0 or args.limit_queries < 0:
        parser.error("Limits must be nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from unifunc3d.data import audit
    if args.require_full_split and (args.visits or args.desc_ids or args.limit_scenes or args.limit_queries):
        parser.error("Full-split mode disallows query/scene filters")
    report = audit(args.dataset_root, args.split, args.decode, args.visits, args.metadata_overrides)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "assets.json", report, pretty=True)
    logging.info("Complete scenes: %d/%d", len(report["complete_scenes"]), len(report["scenes"]))
    logging.info("Complete with matching release fingerprints: %d", len(report["benchmark_compatible_scenes"]))
    for scene in report["scenes"]:
        if not scene["complete"]:
            logging.warning("Skipping %s: %s", scene["visit_id"], scene["errors"])
        elif scene["fingerprint_mismatches"]:
            logging.warning("Skipping %s: fingerprint mismatch %s", scene["visit_id"], scene["fingerprint_mismatches"])
    if args.command == "audit":
        return
    if args.require_full_split and len(report['benchmark_compatible_scenes']) != len(report['scenes']):
        parser.error("Full validation assets/metadata checks failed; inspect assets.json")
    config = Config(**read_json(args.config)) if args.config else Config()
    visits = args.visits or report["benchmark_compatible_scenes"]
    if not visits or set(visits) - set(report["benchmark_compatible_scenes"]):
        parser.error("Requested scenes need complete assets and matching fingerprints; inspect assets.json")
    if args.limit_scenes:
        visits = visits[:args.limit_scenes]
    from unifunc3d.pipeline import Pipeline
    run_info = {"dataset_root": str(args.dataset_root.resolve()), "split": args.split, "visits": visits,
                "limit_queries": args.limit_queries, "desc_ids": args.desc_ids,
                "config": asdict(config), "metadata_overrides": report['metadata_overrides'],
                "require_full_split": args.require_full_split}
    write_json(args.output / "run.json", run_info, pretty=True)
    history_path = args.output / "run_history.json"
    history = read_json(history_path) if history_path.exists() else []
    history.append(run_info)
    write_json(history_path, history, pretty=True)
    with Pipeline(config, args.output) as pipeline:
        pipeline.run(args.dataset_root, args.split, visits, args.limit_queries, args.desc_ids)


if __name__ == "__main__":
    main()
