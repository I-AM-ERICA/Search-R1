#!/usr/bin/env python3
"""Validate reviewed real-search demonstrations; optionally tokenize. Never train."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_r1.policy_sft import validate_dataset, compile_record, fingerprint


def load_capture(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if (manifest["engine"] != "naver" or manifest["official_only"] is not True
            or manifest["require_page_evidence"] is not True or manifest["query_rewriting"] is not False):
        raise ValueError("Unexpected captured search environment")
    plan = json.loads((path / "plan.json").read_text())
    if fingerprint(plan) != manifest["plan_sha256"]:
        raise ValueError("Collection plan changed")
    snapshots = {}
    for key, expected in manifest["snapshot_sha256"].items():
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid snapshot key")
        source = path / "snapshots" / (key + ".json")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Captured evidence changed")
        snapshots[key] = json.loads(source.read_text())
    return plan, snapshots, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--reviews", required=True, help="New reviewed copy of drafts.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", help="Local tokenizer directory; omit for structural review only")
    parser.add_argument("--config", default="configs/policy_search_sft_v1.json")
    args = parser.parse_args()
    destination = Path(args.output_dir)
    if destination.exists():
        parser.error("Use a new output directory")
    plan, snapshots, captured = load_capture(args.capture)
    rows = json.loads(Path(args.reviews).read_text())
    lookup = {c["id"]: c for c in plan["cases"]}
    for row in rows:
        original = lookup.get(row["id"])
        if not original or any(row.get(k) != original.get(k) for k in ("question", "split", "group_id", "agency", "as_of")):
            parser.error("Reviewed identity/split differs from collection plan")
    exclusions = json.loads((Path(__file__).resolve().parents[1] / "evaluations/policy_search_v2/exclusions.json").read_text())
    report = validate_dataset(rows, snapshots, exclusions["questions"], exclusions["urls"])
    report.update(training_started=False, tokenized=False, ready_for_training=False,
                  reviewed_data_sha256=fingerprint(rows), capture_manifest_sha256=fingerprint(captured))
    compiled = []
    if args.tokenizer:
        from transformers import AutoTokenizer
        from policy_r1.config import Config
        from policy_r1.inference import POLICY_SYSTEM_PROMPT
        settings = json.loads(Path(args.config).read_text())
        config = Config(**settings["model_environment"]).validate()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        compiled = [compile_record(row, snapshots, tokenizer, config, POLICY_SYSTEM_PROMPT) for row in rows]
        report.update(tokenized=True, config=settings, prompt_sha256=fingerprint(POLICY_SYSTEM_PROMPT),
                      tokenizer_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in Path(args.tokenizer).iterdir() if p.is_file()})
        train = [r for r in rows if r["split"] == "train"]
        gates = settings["data_gates"]
        recovery = sum(len(r["steps"]) == 2 and r["outcome"] == "supported" for r in train)
        checks = {"train_count": len(train) >= gates["min_train_records"],
                  "val_count": report["splits"]["val"] >= gates["min_val_records"],
                  "reviewed_recovery": recovery >= gates["min_supported_two_search_records"],
                  "single_search": any(len(r["steps"]) == 1 and r["outcome"] == "supported" for r in train),
                  "insufficient_evidence": any(r["outcome"] == "insufficient" for r in train)}
        report["data_gates"] = checks
        report["ready_for_training"] = all(checks.values())
    destination.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        with (destination / (split + ".jsonl")).open("x", encoding="utf-8") as handle:
            for row in compiled:
                if row["split"] == split:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (destination / "reviewed-records.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    report["output_sha256"] = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                               for name in ("train.jsonl", "val.jsonl", "test.jsonl", "reviewed-records.json")}
    source_root = Path(__file__).resolve().parents[1]
    report["compiler_code_sha256"] = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ("policy_r1/policy_sft.py", "policy_r1/observations.py", "policy_r1/paragraph_selection.py",
                     "policy_r1/rollout.py", "policy_r1/protocol.py")}
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
