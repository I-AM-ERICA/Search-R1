#!/usr/bin/env python3
"""Read-only collection audit. Body retrieval is NOT a successful policy answer."""
import argparse
from collections import Counter
import json
from pathlib import Path

from prepare_policy_sft import load_capture


def analyze(path):
    plan, snapshots, manifest = load_capture(path)
    classes, failures, rejected, counts = Counter(), Counter(), Counter(), Counter()
    queries, documents = [], {}
    for key, record in snapshots.items():
        d = record.get("diagnostics", {})
        hits = record.get("hits", [])
        kind = ("provider_error" if record.get("error_code") else
                "body_received" if hits else
                "provider_zero_rows" if d.get("result_count") == 0 else
                "filtered_before_fetch" if d.get("page_fetch_attempts") == 0 else
                "page_attempts_without_body")
        classes[kind] += 1
        failures.update(d.get("page_fetch_failures", {}))
        rejected.update(d.get("rejected", {}))
        counts.update(returned_rows=d.get("result_count", 0),
                      page_attempts=d.get("page_fetch_attempts", 0), returned_bodies=len(hits))
        queries.append({"snapshot_key": key, "query": record["query"], "classification": kind,
                        "body_count": len(hits), "diagnostics": d})
        for h in hits:
            documents[h["id"]] = {"id": h["id"], "title": h["title"], "url": h["url"],
                                  "body_chars": len(h["snippet"]), "semantic_review": "pending"}
    return {"capture": str(Path(path).resolve()), "integrity_verified": True,
            "queries": len(snapshots), "cases": len(plan["cases"]),
            "query_classes": dict(classes), "totals": dict(counts),
            "page_failure_counts": dict(failures), "rejected_candidate_counts": dict(rejected),
            "documents": list(documents.values()), "query_details": queries,
            "ready_for_training": False,
            "limitations": ["Official publisher is not proof of relevance, eligibility or current availability",
                "Old captures without candidate_trace cannot identify failed URLs or individual HTTP statuses",
                "A zero official result is not proof that the program does not exist",
                "This audit is not a learned model evaluation or SFT approval"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze(args.capture)
    with Path(args.output).open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("queries", "query_classes", "totals", "page_failure_counts")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
