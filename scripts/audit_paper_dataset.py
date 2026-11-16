#!/usr/bin/env python3
"""Read-only QA overlap audit; export held-out HotpotQA evaluation candidates. No training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_r1.data import question_key
from policy_r1.prepare import file_sha256, read_parquet_qa


def read_jsonl(path):
    # splitlines() also splits valid Unicode separators inside JSON strings.
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze(raw, prepared, excluded, limit=100):
    if limit < 1:
        raise ValueError('limit must be positive')
    keys = {split: {question_key(r['question']) for r in rows}
            for split, rows in prepared.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if keys[a] & keys[b]:
            raise ValueError('Existing prepared splits overlap')
    known = set().union(*keys.values())
    report = {}
    for split, rows in raw.items():
        questions = set(rows)
        report[split] = dict(unique_questions=len(questions),
            sources=dict(Counter(r['data_source'] for r in rows.values())),
            overlap_with_prepared={s: len(questions & k) for s, k in keys.items()},
            not_in_any_prepared_split=len(questions - known))
    blocked = {question_key(r['question']) for r in excluded}
    candidates = [r for r in prepared['val'] if r['data_source'] == 'hotpotqa'
                  and question_key(r['question']) not in blocked]
    # Stable selection independent of labels, answer length and model performance.
    import hashlib
    candidates.sort(key=lambda r: hashlib.sha256(
        ('search-r1-general-eval-v1:' + question_key(r['question'])).encode()).hexdigest())
    return dict(overlap=report, raw_train_test_overlap=len(set(raw['train']) & set(raw['test'])),
                eligible_hotpot_validation=len(candidates), selected_count=min(limit, len(candidates)),
                selection_note='HotpotQA provenance is only a multi-hop candidate indicator; actual need for multiple searches is unverified.'), candidates[:limit]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', required=True)
    p.add_argument('--prepared-dir', required=True)
    p.add_argument('--exclude-cases', required=True, help='Previously inspected evaluation cases JSON')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--limit', type=int, default=100)
    args = p.parse_args()
    destination = Path(args.output_dir)
    if destination.exists():
        p.error('Use a new output directory')
    source, prepared_dir = Path(args.raw_dir), Path(args.prepared_dir)
    manifest = json.loads((prepared_dir / 'manifest.json').read_text())
    prepared = {}
    for split in ('train', 'val', 'test'):
        path = prepared_dir / (split + '.jsonl')
        if file_sha256(path) != manifest['output_sha256'][path.name]:
            raise ValueError('Prepared split differs from recorded manifest: ' + split)
        prepared[split] = read_jsonl(path)
    raw, stats, hashes = {}, {}, {}
    for split in ('train', 'test'):
        path = source / (split + '.parquet')
        hashes[split] = file_sha256(path)
        raw[split], stats[split] = read_parquet_qa(path, split)
    report, cases = analyze(raw, prepared, json.loads(Path(args.exclude_cases).read_text()), args.limit)
    report.update(raw_stats=stats, raw_sha256=hashes,
        same_bytes_as_original_preparation={s: hashes[s] == manifest['inputs'][s+'_sha256'] for s in hashes},
        inputs=dict(raw_dir=str(source), prepared_dir=str(prepared_dir),
                    prepared_manifest_sha256=file_sha256(prepared_dir/'manifest.json'),
                    excluded_cases_sha256=file_sha256(args.exclude_cases)),
        training_started=False, model_loaded=False, network_requests=0,
        limitations=['Normalized exact-question overlap, not semantic paraphrase detection.',
                     'Prepared training membership does not mean the PPO150 run consumed every row.',
                     'Evaluation candidates are not SFT trajectories and must never become training data.'])
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (destination/'hotpot-validation-candidates.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
