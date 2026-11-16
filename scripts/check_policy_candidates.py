#!/usr/bin/env python3
"""CPU-only check against an existing local tokenizer. Never load model weights."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_policy_sft import load_capture
from policy_r1.policy_sft import compile_record, validate_dataset, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", default="configs/policy_search_sft_v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Use a new output filename")
    from transformers import AutoTokenizer
    from policy_r1.config import Config
    from policy_r1.inference import POLICY_SYSTEM_PROMPT
    _, snapshots, _ = load_capture(args.capture)
    rows = json.loads(Path(args.reviews).read_text())
    exclusions = json.loads(Path("evaluations/policy_search_v2/exclusions.json").read_text())
    validation = validate_dataset(rows, snapshots, exclusions["questions"], exclusions["urls"])
    settings = json.loads(Path(args.config).read_text())
    config = Config(**settings["model_environment"]).validate()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    results = []
    for row in rows:
        result = {"id": row["id"], "split": row["split"], "outcome": row["outcome"]}
        try:
            compiled = compile_record(row, snapshots, tokenizer, config, POLICY_SYSTEM_PROMPT)
            result.update(status="passed", tokens=len(compiled["input_ids"]),
                action_tokens=sum(compiled["action_mask"]),
                observations=[tokenizer.decode(compiled["input_ids"][s["start"]:s["end"]], skip_special_tokens=True)
                              for s in compiled["spans"] if s["type"] == "observation"])
        except ValueError as error:
            result.update(status="rejected", reason=str(error))
        results.append(result)
        print(row["id"], result["status"], result.get("reason", ""), flush=True)
    report = {"checks": results, "structural_validation": validation, "config": settings,
              "reviewed_candidates_sha256": fingerprint(rows), "training_started": False,
              "ready_for_training": False, "note": "Small review pilot; passing token checks is not dataset sufficiency or model quality"}
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
