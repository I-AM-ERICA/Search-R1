"""Compare exact old/new retrieval on real logged queries, without loading a GPU model."""
import argparse
from collections import deque
import json
from pathlib import Path
import re
import sqlite3
from statistics import median
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_r1.sqlite_search import SQLiteRetriever, STOP_WORDS
from policy_r1.protocol import parse_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2,
                        help="Alternate execution order on repeated measurements; result caching stays disabled")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.queries < 1 or args.repeats < 1:
        parser.error("--queries and --repeats must be positive")
    config = json.loads((args.run_dir / "config.json").read_text())
    with (args.run_dir / "rollouts.jsonl").open() as handle:
        recent = deque(handle, maxlen=256)
    queries = []
    for line in reversed(recent):
        for event in json.loads(line)["events"]:
            if event.get("type") == "search":
                action, query = parse_action(event["text"])
                if action == "search" and query not in queries:
                    queries.append(query)
        if len(queries) >= args.queries:
            break
    queries = queries[:args.queries]
    if not queries:
        raise SystemExit("No logged search queries found")
    old = sqlite3.connect(Path(config["corpus_path"]).resolve().as_uri() + "?mode=ro", uri=True)
    old.execute("PRAGMA cache_size=-65536")
    new = SQLiteRetriever(config["corpus_path"], result_cache_size=0)
    results = []
    top_k = config["top_k"]
    report = {"sqlite_version": sqlite3.sqlite_version, "index_path": str(Path(config["corpus_path"]).resolve()),
              "index_documents": len(new), "top_k": top_k, "repeats": args.repeats,
              "result_cache_enabled": False, "queries": results, "complete": False}
    def write_report():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            temporary.replace(args.output)
    write_report()
    try:
        for index, query in enumerate(queries):
            tokens = list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold())))[:64]
            terms = [term for term in tokens if term not in STOP_WORDS] or tokens
            expression = " OR ".join('"' + term + '"' for term in terms)
            if not expression:
                continue
            def baseline():
                rows = old.execute(
                    "SELECT d.id, d.contents, hits.rank FROM "
                    "(SELECT rowid, rank FROM passages WHERE passages MATCH ? ORDER BY rank LIMIT ?) hits "
                    "JOIN documents d ON d.rowid=hits.rowid ORDER BY hits.rank", (expression, top_k)).fetchall()
                return [{"id": key, "contents": body, "score": -score} for key, body, score in rows]
            runs = []
            for repeat in range(args.repeats):
                measured = {}
                # Every query runs in both orders when repeats >= 2. Do not
                # flush OS caches on a user's server or count LRU hits as speedup.
                order = [("old", baseline), ("new", lambda: new.search(query, top_k))]
                if (index + repeat) % 2:
                    order.reverse()
                for label, operation in order:
                    print(f"Query {index + 1}/{len(queries)}, repeat {repeat + 1}/{args.repeats}: {label}", flush=True)
                    started = perf_counter()
                    hits = operation()
                    measured[label] = (perf_counter() - started, hits)
                runs.append({"old_seconds": measured["old"][0], "new_seconds": measured["new"][0],
                             "identical_results": measured["old"][1] == measured["new"][1],
                             "old_ids": [hit["id"] for hit in measured["old"][1]],
                             "new_ids": [hit["id"] for hit in measured["new"][1]]})
            result = {"query": query, "old_seconds": median(run["old_seconds"] for run in runs),
                      "new_seconds": median(run["new_seconds"] for run in runs),
                      "identical_results": all(run["identical_results"] for run in runs), "runs": runs}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            write_report()
        if not results:
            raise SystemExit("No nonempty lexical queries measured; benchmark is not a pass")
        summary = {"queries": len(results), "all_identical": all(r["identical_results"] for r in results),
                   "old_seconds": sum(r["old_seconds"] for r in results),
                   "new_seconds": sum(r["new_seconds"] for r in results)}
        summary["speedup"] = summary["old_seconds"] / max(summary["new_seconds"], 1e-9)
        summary["faster_queries"] = sum(r["new_seconds"] < r["old_seconds"] for r in results)
        summary["search_improved"] = summary["all_identical"] and summary["speedup"] > 1.0
        report.update(summary=summary, complete=True)
        write_report()
        print(json.dumps(summary, indent=2), flush=True)
        if not summary["all_identical"]:
            raise SystemExit("Results differ; do not resume training with the new retriever yet")
        if not summary["search_improved"]:
            raise SystemExit("Search was not faster in this sample; review the report before resuming")
    finally:
        old.close()
        new.close()


if __name__ == "__main__":
    main()
