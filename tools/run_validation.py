#!/usr/bin/env python3
"""Supervise a resumable full validation run and evaluate it automatically."""
import argparse
import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_lib.common import read_json, write_json, reference_rows, compare_row
from evaluate_inference import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/scenefun3d')
    parser.add_argument('--metadata-overrides', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = read_json(ROOT / 'datasets/val.json')
    expected_queries = sum(len(v['desc_ids']) for v in expected.values())
    status = {'state': 'preflight', 'supervisor_pid': os.getpid(),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'expected_scenes': len(expected), 'expected_queries': expected_queries,
              'model': read_json(args.config)['qwen_model']}

    def save_status():
        status['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out / 'status.json', status, pretty=True)

    save_status()
    try:
        # Validate every author's published score before comparing new inference.
        baseline = evaluate(args.dataset_root, 'val', ROOT / 'predictions/8b_val', args.metadata_overrides)
        reference = reference_rows(read_json(ROOT / 'reference_results/8b_val.json'))
        for row in baseline['per_query']:
            compare_row(row['ours'], reference[(row['visit_id'], row['desc_id'])], row['desc_id'])
        if not baseline['complete_split']:
            raise RuntimeError('Reference evaluation does not cover the full split')
        write_json(out / 'reference_check.json', {'verified_queries': len(baseline['per_query']),
                   'percent': baseline['percent']['ours'], 'metadata_overrides': baseline['metadata_overrides']}, pretty=True)
        sources = [*ROOT.glob('unifunc3d/*.py'), ROOT / 'infer.py', ROOT / 'evaluate_inference.py',
                   Path(__file__).resolve(), args.config.resolve()]
        write_json(out / 'source_snapshot.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in sources}, pretty=True)
        command = [sys.executable, '-u', 'infer.py', 'run', '--require-full-split',
                   '--config', str(args.config.resolve()), '--output', str(out),
                   '--dataset-root', str(args.dataset_root.resolve())]
        if args.metadata_overrides:
            command += ['--metadata-overrides', str(args.metadata_overrides.resolve())]
        env = dict(os.environ, OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='8',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (out / 'run.log').open('a', buffering=1) as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            status.update(state='running', inference_pid=process.pid, command=command)
            last_evaluated = -1
            last_eval_time = time.monotonic()
            while True:
                counts = {p.stem: len(read_json(p)) for p in (out / 'predictions').glob('*.json')}
                status.update(completed_queries=sum(counts.values()), scene_query_counts=counts,
                              completed_scenes=sum(counts.get(k, 0) == len(v['desc_ids']) for k, v in expected.items()))
                save_status()
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'Inference exited with code {code}; inspect run.log and resume the same command')
                    break
                if status['completed_queries'] and status['completed_queries'] != last_evaluated and time.monotonic() - last_eval_time >= 300:
                    result = evaluate(args.dataset_root, 'val', out / 'predictions', args.metadata_overrides)
                    write_json(out / 'evaluation_progress.json', result, pretty=True)
                    last_evaluated = result['queries']
                    last_eval_time = time.monotonic()
                time.sleep(15)
        result = evaluate(args.dataset_root, 'val', out / 'predictions', args.metadata_overrides)
        write_json(out / 'evaluation.json', result, pretty=True)
        if not result['complete_split']:
            raise RuntimeError('Inference finished without full split coverage')
        paper = {'AP50': 23.82, 'AP25': 44.04, 'AR50': 46.07, 'AR25': 55.51, 'mIoU': 20.92}
        comparison = {'model': status['model'], 'queries': result['queries'], 'scenes': result['scenes'],
                      'ours_percent': result['percent']['ours'], 'released_8b_percent': result['percent']['released_8b'],
                      'paper_8b_percent': paper,
                      'delta_vs_paper_percentage_points': {k: round(result['percent']['ours'][k] - v, 4) for k, v in paper.items()},
                      'metadata_overrides': result['metadata_overrides'],
                      'note': 'RynnBrain backbone substitution with supplemented prompts; not identical-checkpoint reproduction.'}
        write_json(out / 'comparison.json', comparison, pretty=True)
        status.update(state='completed', completed_queries=result['queries'], completed_scenes=result['scenes'],
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save_status()
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        save_status()
        raise


if __name__ == '__main__':
    main()
