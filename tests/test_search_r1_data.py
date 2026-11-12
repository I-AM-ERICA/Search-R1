import gzip
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_r1.config import Config
from policy_r1.data import load_datasets, read_jsonl, question_key
from policy_r1.prepare import prepare_qa
from policy_r1.protocol import score_answer
from policy_r1.sqlite_search import SQLiteRetriever, build_sqlite_index


def parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def qa(question, answer="gold", source="nq"):
    return {"id": "reused-source-id", "question": question, "golden_answers": [answer],
            "data_source": source, "prompt": [{"content": "Do not copy this prompt"}]}


def test_real_parquet_split_dedup_and_test_isolation(tmp_path):
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    parquet(train, [qa(f"Question {i}?") for i in range(10)] +
            [qa("QUESTION 0", "alias", "hotpotqa"), qa("test question", "train-only alias")])
    parquet(test, [qa("Test question?", "test answer"), qa(" TEST QUESTION ", "test alias")])
    output = tmp_path / "prepared"
    report = prepare_qa(train, test, output, validation_size=2)
    assert report["counts"] == {"train": 8, "val": 2, "test": 1}
    assert report["train_test_overlap_removed_from_train"] == 1
    rows = {name: read_jsonl(output / f"{name}.jsonl") for name in report["counts"]}
    assert rows["test"][0]["answers"] == ["test answer", "test alias"]
    all_train = rows["train"] + rows["val"]
    assert next(r for r in all_train if question_key(r["question"]) == "question0")["answers"] == ["gold", "alias"]
    assert all("prompt" not in r for split in rows.values() for r in split)
    assert len({r["id"] for split in rows.values() for r in split}) == 11
    assert prepare_qa(train, test, output, validation_size=2) == report
    second = tmp_path / "second"
    prepare_qa(train, test, second, validation_size=2)
    assert (output / "val.jsonl").read_bytes() == (second / "val.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_qa(train, test, output, validation_size=3)
    (output / "train.jsonl").write_text("tampered")
    with pytest.raises(FileExistsError):
        prepare_qa(train, test, output, validation_size=2)


def test_nested_reward_target_and_invalid_question(tmp_path):
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    parquet(train, [{"question": f"q{i}", "reward_model": {"ground_truth": {"target": ["answer"]}}}
                    for i in range(3)])
    parquet(test, [qa("test")])
    report = prepare_qa(train, test, tmp_path / "ok", validation_size=1)
    assert report["counts"]["train"] == 2
    parquet(train, [qa(" ")])
    with pytest.raises(ValueError, match="question"):
        prepare_qa(train, test, tmp_path / "bad", validation_size=1)
    assert not (tmp_path / "bad").exists()


def test_streamed_gzip_index_search_and_data_loader(tmp_path):
    corpus, index = tmp_path / "wiki.jsonl.gz", tmp_path / "wiki.sqlite"
    docs = [{"id": "1", "contents": "Paris is the capital of France."},
            {"id": "2", "text": "Berlin is the capital of Germany.", "title": "Berlin"}]
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        for row in docs:
            handle.write(json.dumps(row) + "\n")
    info = build_sqlite_index(corpus, index, batch_size=1)
    assert info["documents"] == 2
    assert build_sqlite_index(corpus, index) == info
    retriever = SQLiteRetriever(index)
    assert retriever.search('Paris " OR NOT {unknown}:*')[0]["id"] == "1"
    assert retriever.search("Berlin")[0]["id"] == "2"
    assert retriever.search("qqqqnonsense") == []
    assert retriever.search("\" * -") == []
    assert "1" in retriever and "missing" not in retriever
    retriever.close()
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(
            {"id": split, "question": split, "answers": ["Paris"], "evidence_ids": ["1"]}) + "\n")
    config = Config(corpus_path=str(index), retriever="sqlite_bm25", allow_demo=False,
                    **{f"{split}_path": str(tmp_path / f"{split}.jsonl") for split in ("train", "val", "test")})
    documents, train, val = load_datasets(config)
    assert len(documents) == 2 and len(train) == len(val) == 1
    documents.close()
    with pytest.raises(ValueError, match="겹칩니다"):
        load_datasets(replace(config, test_path=config.train_path))


def test_failed_index_is_not_published_and_does_not_replace_existing(tmp_path):
    source, index = tmp_path / "bad.jsonl", tmp_path / "bad.sqlite"
    source.write_text('{"id":1,"contents":"a"}\n{"id":1,"contents":"b"}\n')
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        build_sqlite_index(source, index, batch_size=1)
    assert not index.exists()
    source.write_text('{"id":1,"contents":"a"}\n')
    build_sqlite_index(source, index)
    source.write_text('{"id":1,"contents":"changed"}\n')
    with pytest.raises(FileExistsError):
        build_sqlite_index(source, index)
    retriever = SQLiteRetriever(index)
    assert retriever.search("a")[0]["contents"] == "a"
    retriever.close()


def test_english_em_preserves_word_boundaries_and_removes_articles():
    assert score_answer("The Eiffel Tower.", ["Eiffel Tower"], normalization="search_r1")["exact_match"] == 1
    assert score_answer("Newark", ["New Ark"], normalization="search_r1")["exact_match"] == 0
