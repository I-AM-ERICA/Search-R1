import json
import random
import sqlite3
import sys

import pytest

from policy_r1.sqlite_search import build_sqlite_index, SQLiteRetriever
from scripts import benchmark_search


def test_optimized_search_preserves_exact_ranked_results_and_cache_isolation(tmp_path):
    rng = random.Random(42)
    corpus = tmp_path / "corpus.jsonl"
    words = ["world", "john", "city", "university", "history", "science"]
    corpus.write_text("\n".join(json.dumps({"id": str(i), "contents": " ".join(
        rng.choices(words, k=rng.randrange(5, 40)))}) for i in range(300)))
    database = tmp_path / "index.sqlite"
    build_sqlite_index(corpus, database)
    retriever = SQLiteRetriever(database, result_cache_size=2)
    try:
        for query in ["world john", "science university", "city history", "missing", "world", "the"]:
            expression = " OR ".join(f'"{word}"' for word in query.split())
            expected = retriever.connection.execute(
                "SELECT d.id, d.contents, h.rank FROM "
                "(SELECT rowid, rank FROM passages WHERE passages MATCH ? ORDER BY rank LIMIT 3) h "
                "JOIN documents d ON d.rowid=h.rowid ORDER BY h.rank", (expression,)).fetchall()
            actual = retriever.search(query)
            assert actual == [{"id": key, "contents": text, "score": -rank}
                              for key, text, rank in expected]
        original = retriever.search("world john")
        original[0]["contents"] = "modified by caller"
        assert retriever.search("world john")[0]["contents"] != "modified by caller"
        assert retriever.cache_hits >= 1
        assert len(retriever._cache) <= 2
        assert retriever.search("!!!") == []
        assert retriever.search("world", 0) == []
        assert retriever.search_seconds > 0
    finally:
        retriever.close()
    # A persisted custom ranking function must not be silently replaced.
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO passages(passages, rank) VALUES('rank', 'bm25(2.0)')")
    custom = SQLiteRetriever(database)
    try:
        assert custom._scorer == "rank"
        assert custom.search("world")
    finally:
        custom.close()


@pytest.mark.parametrize("mismatch", [False, True])
def test_benchmark_repeated_comparison_and_report(tmp_path, monkeypatch, mismatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"p1","contents":"Paris France capital city"}\n')
    index = tmp_path / "index.sqlite"
    build_sqlite_index(corpus, index)
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"corpus_path": str(index), "top_k": 3}))
    (run / "rollouts.jsonl").write_text(json.dumps({"events": [
        {"type": "search", "text": "<search>Paris</search>"}]}))
    output = run / "benchmark.json"
    monkeypatch.setattr(sys, "argv", ["benchmark_search.py", "--run-dir", str(run),
                                     "--queries", "1", "--repeats", "2", "--output", str(output)])
    # Old then new; new then old. Avoid wall-clock noise in verdict tests.
    ticks = iter([0, 2, 10, 11, 20, 21, 30, 32])
    monkeypatch.setattr(benchmark_search, "perf_counter", lambda: next(ticks))
    if mismatch:
        original = SQLiteRetriever.search
        def different(self, *args):
            result = original(self, *args)
            result[0]["score"] += 1
            return result
        monkeypatch.setattr(SQLiteRetriever, "search", different)
        with pytest.raises(SystemExit, match="Results differ"):
            benchmark_search.main()
    else:
        benchmark_search.main()
    report = json.loads(output.read_text())
    assert report["complete"] and report["result_cache_enabled"] is False
    assert report["summary"]["speedup"] == 2
    assert report["summary"]["all_identical"] is (not mismatch)
    assert report["summary"]["search_improved"] is (not mismatch)
    assert len(report["queries"][0]["runs"]) == 2


def test_empty_lexical_queries_cannot_pass_benchmark(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"p1","contents":"Paris France"}\n')
    index = tmp_path / "index.sqlite"
    build_sqlite_index(corpus, index)
    (tmp_path / "config.json").write_text(json.dumps({"corpus_path": str(index), "top_k": 3}))
    (tmp_path / "rollouts.jsonl").write_text(json.dumps({"events": [
        {"type": "search", "text": "<search>!!!</search>"}]}))
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(sys, "argv", ["benchmark_search.py", "--run-dir", str(tmp_path), "--output", str(output)])
    with pytest.raises(SystemExit, match="No nonempty lexical"):
        benchmark_search.main()
    assert json.loads(output.read_text())["complete"] is False
