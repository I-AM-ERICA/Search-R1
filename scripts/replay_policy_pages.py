#!/usr/bin/env python3
"""Replay recorded SerpAPI candidate URLs; optionally fetch only unread HTML.

No provider request, query invention, attachment conversion or model loading.
Provider ordering is frozen. Any newly read page is labeled live, so the output
is an extraction/selection ablation, not a fully frozen historical-web benchmark.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_policy_sft import load_capture
from scripts.collect_policy_curriculum import write_new
from policy_r1.policy_sft import fingerprint
from policy_r1.web_search import SerpAPIRetriever
from policy_r1.official_pages import fetch_public_page, PageUnavailable, OFFICIAL_REGISTRY


def replay(source, destination, *, fetch_missing=False, max_live_pages=12, fetcher=fetch_public_page):
    plan, snapshots, original_manifest = load_capture(source)
    destination = Path(destination)
    known, failed = {}, {}
    for snapshot in snapshots.values():
        diagnostics = snapshot.get("diagnostics", {})
        trace = diagnostics.get("candidate_trace", [])
        # Older captures without full candidate order cannot support this replay.
        if sorted(r["rank"] for r in trace) != list(range(1, diagnostics.get("result_count", 0) + 1)):
            raise ValueError("Incomplete candidate trace; do not fabricate missing search results")
        for h in snapshot["hits"]:
            if h["url"] in known and known[h["url"]]["text"] != h["snippet"]:
                raise ValueError("Page contents changed within source capture")
            known[h["url"]] = {"text": h["snippet"], "origin": "recorded_body"}
        for row in trace:
            if row["status"] == "page_unavailable":
                failed[row["url"]] = {"error": row["failure_reason"],
                                       "details": row.get("failure_details", {}), "origin": "recorded_failure"}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "snapshots").mkdir()
    pages, live = {}, 0

    def page(url, timeout):
        nonlocal live
        if url not in pages:
            if url in known:
                pages[url] = known[url]
            elif url in failed:
                pages[url] = failed[url]
            else:
                if not fetch_missing:
                    raise RuntimeError("Unrecorded HTML page; use explicit fetch-missing-pages option")
                if live >= max_live_pages:
                    raise RuntimeError("Live HTML budget exceeded")
                live += 1
                record = {"origin": "live_missing_page", "captured_at": datetime.now(timezone.utc).isoformat()}
                try:
                    record["text"] = fetcher(url, timeout=timeout)
                except PageUnavailable as error:
                    record.update(error=str(error), details=error.details)
                pages[url] = record
            # Write progressively so failed/partial runs also retain live fetch evidence.
            (destination / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2))
        record = pages[url]
        if "error" in record:
            error = PageUnavailable(record["error"])
            error.details = record.get("details", {})
            raise error
        return record["text"]

    files, summary = {}, []
    for key, old in snapshots.items():
        trace = sorted(old["diagnostics"]["candidate_trace"], key=lambda r: r["rank"])
        payload = {"web_results": [{"link": r["url"], "title": r["title"]} for r in trace],
                   "search_parameters": {"query": old["query"]}}

        def memory_response(request, timeout):
            if parse_qs(urlsplit(request.full_url).query)["query"] != [old["query"]]:
                raise ValueError("Replay query mismatch")
            return io.BytesIO(json.dumps(payload).encode())

        backend = SerpAPIRetriever("offline-no-api-call", engine="naver", cache_size=0,
            require_page_evidence=True, official_only=True, opener=memory_response, page_fetcher=page)
        hits = backend.search(old["query"], old["top_k"], timeout=30)
        record = {"query": old["query"], "top_k": old["top_k"], "hits": hits,
                  "diagnostics": backend.last_diagnostics, "provider_replayed": True,
                  "provider_captured_at": old["captured_at"],
                  "captured_at": datetime.now(timezone.utc).isoformat()}
        path = destination / "snapshots" / (key + ".json")
        write_new(path, record)
        files[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary.append({"query": old["query"], "before_bodies": len(old["hits"]), "after_bodies": len(hits)})
        print(old["query"], len(old["hits"]), "->", len(hits), flush=True)
    write_new(destination / "plan.json", plan)
    write_new(destination / "manifest.json", {
        "schema_version": 1, "purpose": "candidate_selection_replay_not_model_evaluation",
        "plan_sha256": fingerprint(plan), "engine": "naver", "official_only": True,
        "require_page_evidence": True, "query_rewriting": False, "top_k": 3,
        "snapshot_sha256": files, "official_registry": OFFICIAL_REGISTRY,
        "source_manifest_sha256": fingerprint(original_manifest), "source_capture": str(Path(source).resolve()),
        "unique_provider_requests": 0, "live_missing_page_requests": live, "ready_for_training": False,
        "records": len(plan["cases"]), "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__).resolve().parents[1] / "policy_r1").glob("*.py")}})
    write_new(destination / "comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fetch-missing-pages", action="store_true")
    parser.add_argument("--max-live-pages", type=int, default=12)
    args = parser.parse_args()
    if args.max_live_pages < 0:
        parser.error("Page budget must be nonnegative")
    replay(args.source, args.output_dir, fetch_missing=args.fetch_missing_pages, max_live_pages=args.max_live_pages)


if __name__ == "__main__":
    main()
