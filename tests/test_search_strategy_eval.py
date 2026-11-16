import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_ppo_run import analyze
from compare_search_strategy import FrozenSearch, RetrievalFailure, run_one, summarize, make_reviews
from grade_search_strategy import aggregate, paired_stats
from test_summary_comparison import ToggleActor
from test_rollout import CharacterTokenizer
from analyze_search_behavior import behavior
from policy_r1.config import Config
from policy_r1.protocol import score_answer


class AdapterActor(ToggleActor):
    def set_adapter(self, name):
        self.active_adapter = name


class Backend:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        return [{"id": "doc", "contents": "공식 조건은 5인 미만", "url": "https://www.semas.or.kr/rule"}]


def test_sft_variant_uses_sft_adapter_then_restores_default(tmp_path):
    tokenizer=CharacterTokenizer()
    class TracedActor(AdapterActor):
        def set_adapter(self,name):
            super().set_adapter(name)
            self.adapter_calls=getattr(self,'adapter_calls',[])+[name]
    actor=TracedActor(['<answer>가상 답변</answer>'],tokenizer)
    row=run_one(actor,tokenizer,Config(max_seq_length=3000),FrozenSearch(Backend(),tmp_path/'sft'),
                {'id':'c','question':'가상 질문'},'sft','raw',None,42)
    assert actor.adapter_calls==['sft','default']
    assert actor.enabled and row['variant']=='sft' and row['format_complete']


def test_same_query_freezes_results_and_replay_never_returns_empty_on_miss(tmp_path):
    backend = Backend()
    frozen = FrozenSearch(backend, tmp_path/"record")
    hits = frozen.search("조건", 3)
    hits[0]["contents"] = "changed"
    assert frozen.search("조건", 3)[0]["contents"] == "공식 조건은 5인 미만"
    assert backend.queries == ["조건"]
    replay = FrozenSearch(None, tmp_path/"replayed", replay=tmp_path/"record")
    assert replay.search("조건", 3) == frozen.search("조건", 3)
    with pytest.raises(RetrievalFailure, match="REPLAY_MISS"):
        replay.search("다른 질의", 3)
    limited = FrozenSearch(backend, tmp_path/"limited", budget=1)
    limited.search("first", 3)
    with pytest.raises(RetrievalFailure, match="BUDGET"):
        limited.search("second", 3)


def test_live_model_queries_are_not_replaced_by_gold_and_second_search_is_traced(tmp_path):
    tokenizer = CharacterTokenizer()
    actor = AdapterActor(["<search>자격</search>", "<search>근로자 기준</search>", "<answer>5인 미만</answer>"], tokenizer)
    backend = Backend()
    frozen = FrozenSearch(backend, tmp_path/"snapshots")
    case = {"id": "c", "question": "지원 조건?", "answers": ["5인 미만"], "review_points": ["secret gold rubric"]}
    row = run_one(actor, tokenizer, Config(max_seq_length=3000, max_new_tokens=100), frozen, case,
                  "base", "raw", "same prompt", 42)
    assert backend.queries == ["자격", "근로자 기준"]
    assert row["exact_match"] == 1 and row["second_model_search"]
    assert row["provider_requests"] == 2 and actor.enabled
    assert row["mask_check"]["generated_count_matches_loss_mask"]
    assert all(a["visible_ids"] == ["doc"] for a in row["search_actions"])
    reviews = make_reviews([row], [case], 42)
    assert "variant" not in reviews[0]
    assert summarize([row])["base"]["human_fully_correct_answers"]["rate"] is None
    assert summarize([row], reviews)["base"]["human_relevant_search_success"]["rate"] is None


def test_forced_first_empty_is_explicit_and_does_not_spend_a_provider_call(tmp_path):
    tokenizer = CharacterTokenizer()
    actor = AdapterActor(["<search>one</search>", "<search>two</search>", "<answer>ok</answer>"], tokenizer)
    backend = Backend()
    row = run_one(actor, tokenizer, Config(max_seq_length=3000), FrozenSearch(backend,tmp_path/"snap"),
                  {"id":"c","question":"q"}, "trained", "raw", None, 42, first_empty=True)
    assert backend.queries == ["two"]
    assert row["intervention"] == "first_empty"
    assert row["search_actions"][0]["visible_ids"] == []
    assert row["provider_requests"] == 1
    metrics=behavior([row])['trained']
    assert metrics['first_empty_questions']==1 and metrics['second_search_after_first_empty']==1
    assert metrics['changed_second_queries']==1
    assert metrics['second_nonempty_after_first_empty_PROXY']==1


def test_environment_failure_is_not_counted_as_empty_result_recovery():
    row=dict(variant='base',environment_error='TIMEOUT',search_actions=[{'query':'q','returned_hits':[]}],
             format_complete=False,finish_reason='retrieval_error')
    result=behavior([row])['base']
    assert result['environment_errors']==1
    assert result['first_empty_questions']==0


def test_assisted_provider_retry_is_not_counted_as_second_model_search(tmp_path):
    tokenizer = CharacterTokenizer()
    actor = AdapterActor(["<search>소상공인시장진흥공단 정책자금 공통 지원 자격</search>", "<answer>5인 미만</answer>"], tokenizer)
    class EmptyFirst(Backend):
        def search(self, query, top_k):
            hits = super().search(query, top_k)
            return [] if len(self.queries) == 1 else hits
    backend = EmptyFirst()
    row = run_one(actor, tokenizer, Config(max_seq_length=3000), FrozenSearch(backend,tmp_path/"snap"),
        {"id":"c","question":"소상공인시장진흥공단 공통 자격"}, "trained", "assisted", None, 42)
    assert row["provider_requests"] == 2
    assert len(row["search_actions"]) == 1 and not row["second_model_search"]


def test_log_audit_preserves_missingness_and_rejects_conflicting_runs(tmp_path):
    d=tmp_path/"a";d.mkdir()
    rows=[{"step":1,"reward":0,"action_tokens":80}, {"step":3,"reward":1,"action_tokens":160}]
    (d/"metrics.jsonl").write_text("\n".join(map(json.dumps,rows)))
    result=analyze([d],until=3,batch_size=8)
    assert result["missing_steps"] == [2]
    assert result["overall"]["reward"]["mean"] == .5
    assert result["answer_action_tokens_from_available_traces"]["mean"] is None
    assert not result["trace_coverage_complete"]
    other=tmp_path/"b";other.mkdir()
    (other/"metrics.jsonl").write_text(json.dumps({"step":1,"reward":1}))
    with pytest.raises(ValueError,match="Conflicting"):
        analyze([d,other],until=3)


def test_current_reward_does_not_reward_format_or_search_but_rejects_explanatory_answer():
    gold=["5인 미만"]
    assert score_answer("5인 미만",gold,searches=0)["reward"] == 1
    assert score_answer("5인 미만",gold,searches=2)["reward"] == 1
    assert score_answer("조건은 5인 미만입니다",gold,searches=1)["reward"] == 0
    assert score_answer("정책 안내입니다",gold,searches=1)["reward"] == 0
    assert score_answer("",gold,searches=2)["reward"] == 0


def test_grades_require_explicit_source_judgement_and_paired_analysis():
    rows=[]
    for variant in ["base","trained"]:
        rows.append({"variant":variant,"case_id":"c","question":"q","answer":"a",
          "finish_reason":"answer","format_complete":True,"nonempty_visible_search_proxy":True,
          "second_model_search":True,"search_actions":[{"query":"q1"},{"query":"q2"}]})
    reviews=make_reviews(rows,[{"id":"c"}],42)
    assert aggregate(rows,reviews)["paired"]["base_vs_trained_fully_correct"]["n"] == 0
    for row in rows:
        review=next(r for r in reviews if r["review_id"] == row["review_id"])
        review["accuracy_0_to_2"]=2 if row["variant"] == "trained" else 0
        review["coverage_0_to_2"]=2
        for i,g in enumerate(review["search_actions"]):
            g.update(query_quality_0_to_2=2,relevant_visible_evidence=bool(i),
                     sufficient_visible_evidence=bool(i),new_useful_evidence=bool(i))
    result=aggregate(rows,reviews)
    assert result["paired"]["base_vs_trained_fully_correct"]["wins"] == 1
    assert result["by_variant"]["trained"]["human_additional_search_success_given_first_insufficient"]["rate"] == 1
    assert paired_stats([(0,1),(1,0)])["exact_mcnemar_p"] == 1
    assert paired_stats([(1,1),(0,0)])["paired_bootstrap_95_percent"] is None
    reviews[0]["accuracy_0_to_2"] = True
    with pytest.raises(ValueError):
        aggregate(rows,reviews)


def test_blank_no_search_review_is_not_mistaken_for_completed_human_review():
    row={"variant":"base","case_id":"c","question":"q","answer":"", "finish_reason":"invalid_action",
         "format_complete":False,"nonempty_visible_search_proxy":False,"second_model_search":False,"search_actions":[]}
    reviews=make_reviews([row],[{"id":"c"}],42)
    result=aggregate([row],reviews)
    assert result["by_variant"]["base"]["human_relevant_search_success"]["n"] == 0
