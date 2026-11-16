#!/usr/bin/env python3
"""Capture real searches for teacher authoring. No GPU/model, no training.

Planned queries are candidate demonstrations, NOT claims of learned/adaptive
search. Both candidates are collected for review; reviewers select 1 or 2 steps.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_r1.policy_sft import fingerprint
from policy_r1.web_search import SerpAPIRetriever, SearchUnavailable
from policy_r1.official_pages import OFFICIAL_REGISTRY


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def collect(plan, search, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "snapshots").mkdir()
    snapshots, drafts, files = {}, [], {}
    write_new(destination / "plan.json", plan)
    for case in plan["cases"]:
        steps = []
        for query in case["candidate_queries"]:
            key = fingerprint([query, 3])
            if key not in snapshots:
                record = {"query": query, "top_k": 3,
                          "captured_at": datetime.now(timezone.utc).isoformat()}
                try:
                    record["hits"] = search.search(query, 3, timeout=search.timeout)
                    record["diagnostics"] = search.last_diagnostics
                except SearchUnavailable as error:
                    record.update(hits=[], error_code=error.code)
                snapshots[key] = record
                path = destination / "snapshots" / (key + ".json")
                write_new(path, record)
                files[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            steps.append({"query": query, "snapshot_key": key, "reason_for_next_search": ""})
            print(f"{case['id']}: {len(snapshots[key]['hits'])} official bodies; "
                  f"{snapshots[key].get('error_code', 'captured')}", flush=True)
        drafts.append({k: v for k, v in case.items() if k != "candidate_queries"} | {
            "steps": steps, "answer": "", "claims": [], "limitations": [], "outcome": None,
            "review": {"status": "pending", "reviewer": "", "evidence_checked": False,
                       "coverage_checked": False, "notes": "", "content_sha256": ""}})
    write_new(destination / "drafts.json", drafts)
    write_new(destination / "manifest.json", {"schema_version": 1, "purpose": "teacher_candidates_not_training_ready",
        "plan_sha256": fingerprint(plan), "engine": search.engine, "official_only": True,
        "require_page_evidence": True, "query_rewriting": False, "top_k": 3,
        "snapshot_sha256": files, "official_registry": OFFICIAL_REGISTRY,
        "unique_provider_requests": len(snapshots), "records": len(drafts),
        "ready_for_training": False, "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve().parents[1] / "policy_r1").glob("*.py")}})
    print("CAPTURE COMPLETE:", destination)
    print("No training performed. Answers and semantic review are still pending.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="evaluations/policy_search_v2/collection_plan.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-requests", type=int, default=36)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    cases = plan["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        parser.error("Need nonempty unique case IDs")
    if any(not 1 <= len(c["candidate_queries"]) <= 2 for c in cases):
        parser.error("Each case must have one or two candidate queries")
    queries = {q for c in cases for q in c["candidate_queries"]}
    if len(queries) > args.max_requests:
        parser.error("Plan exceeds explicit API request budget")
    search = SerpAPIRetriever(engine="naver", timeout=args.timeout, cache_size=128,
                              require_page_evidence=True, official_only=True)
    collect(plan, search, args.output_dir)


if __name__ == "__main__":
    main()
