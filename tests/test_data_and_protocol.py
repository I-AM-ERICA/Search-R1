# 역할: 정책 데이터 분할·검색·한국어 정답 보상·태그 파싱의 실패 조건을 검증합니다.
import json
from dataclasses import replace

import pytest

from policy_r1.config import load_config
from policy_r1.data import load_datasets
from policy_r1.protocol import parse_action, score_answer
from policy_r1.retrieval import BM25Retriever


def test_demo_retrieval_and_split():
    config = load_config()
    with pytest.warns(UserWarning):
        corpus, train, val = load_datasets(config)
    hits = BM25Retriever(corpus).search("청년 배움지원금", 2)
    assert hits[0]["id"] == "demo-youth"
    assert not {r["id"] for r in train} & {r["id"] for r in val}
    assert BM25Retriever(corpus).search("zzzznotfound") == []
    with pytest.raises(ValueError, match="가상 예제"):
        load_datasets(replace(config, allow_demo=False))


def test_leaking_validation_rejected(tmp_path):
    config = load_config()
    row = json.loads(open(config.train_path, encoding="utf-8").readline())
    row["id"] = "new-id-same-question"
    path = tmp_path / "val.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="겹칩니다"):
        load_datasets(replace(config, val_path=str(path)))


def test_reward_and_strict_protocol():
    assert parse_action("<think>확인</think><search>청년</search>") == ("search", "청년")
    assert parse_action("<answer>20만 원</answer>") == ("answer", "20만 원")
    for text in ["<answer></answer>", "<answer>20만 원", "답은 20만 원",
                 "<information><answer>20만 원</answer></information>",
                 "<answer>오답</answer><answer>정답</answer>"]:
        assert parse_action(text)[0] == "invalid"
    assert score_answer("２０만 원.", ["20만원"])["exact_match"] == 1
    assert score_answer("120만원", ["20만원"])["exact_match"] == 0
    assert score_answer("", ["20만원"])["reward"] == 0
    assert score_answer("20만원", ["20만원"], searches=2, search_cost=0.1)["reward"] == pytest.approx(0.8)
