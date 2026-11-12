# 역할: 0 보상 정체를 재현하고 형식 SFT의 실제 gradient·가중치 변경과 데이터 경계를 검증합니다.
from dataclasses import replace
import json

import pytest
import torch

from policy_r1.config import Config
from policy_r1 import trainer
from policy_r1.models import action_log_probs
from policy_r1.retrieval import BM25Retriever
from policy_r1.rollout import Trajectory
from policy_r1.warmup import build_protocol_examples
from test_ppo import tiny_models
from test_rollout import CharacterTokenizer
from test_trainer import TinyTokenizer


def make_experiment(tmp_path, monkeypatch, format_reward=0.0):
    monkeypatch.setattr(trainer, "load_models", lambda *args: (TinyTokenizer(), *tiny_models()))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test double")
    config = Config(load_in_4bit=False, rollout_batch_size=2, ppo_epochs=1,
                    total_updates=5, max_zero_signal_updates=2, eval_every=5, save_every=5,
                    format_reward=format_reward, output_dir=str(tmp_path / "run"))
    with pytest.warns(UserWarning):
        experiment = trainer.PPOExperiment(config)
    return experiment


def invalid_rollout(*args, **kwargs):
    return Trajectory([1, 5, 6, 7, 8, 9], [False, False, True, True, True, True],
                      "", 0, "generation_limit", [{"type": "invalid", "text": 'assistant -> tool/python search("policy")'}])


def test_all_zero_reward_reproduces_frozen_parameters_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "collect_rollout", invalid_rollout)
    experiment = make_experiment(tmp_path, monkeypatch)
    weights = {n: p.detach().clone() for n, p in experiment.actor.named_parameters() if p.requires_grad}
    batch = experiment._collect_batch()
    for item in batch:
        assert item["score"]["reward"] == 0
        assert torch.count_nonzero(item["old_values"]) == 0
        assert torch.count_nonzero(item["advantages"]) == 0
        assert torch.count_nonzero(item["returns"]) == 0
        assert item["reference_kl"] == 0
    metrics = experiment._update(batch)
    assert all(value == 0 for value in metrics.values())
    assert all(torch.equal(weights[n], p) for n, p in experiment.actor.named_parameters() if n in weights)
    with pytest.raises(RuntimeError, match="2회 연속"):
        experiment.train()
    saved = [json.loads(line) for line in (experiment.output / "metrics.jsonl").read_text().splitlines()]
    assert saved[-1]["zero_signal_updates"] == 2
    assert saved[-1]["finish_reasons"] == {"generation_limit": 2}
    traces = [json.loads(line) for line in (experiment.output / "rollouts.jsonl").read_text().splitlines()]
    assert len(traces) == 4
    assert {row["batch_index"] for row in traces} == {0, 1}
    assert not list(experiment.output.glob("checkpoint-*"))


def test_format_bonus_alone_cannot_help_unparseable_output(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "collect_rollout", invalid_rollout)
    experiment = make_experiment(tmp_path, monkeypatch, format_reward=0.1)
    assert all(item["score"]["reward"] == 0 for item in experiment._collect_batch())


def test_warmup_updates_actor_not_reference_or_critic_and_uses_train_only(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch)
    example = Trajectory([1, 5, 6, 7, 8, 9], [False, False, True, True, False, True],
                         "gold", 1, "answer", [])
    def build(tokenizer, retriever, rows, config):
        assert rows is experiment.train_data
        assert not {r["id"] for r in rows} & {r["id"] for r in experiment.val_data}
        return [example]
    monkeypatch.setattr(trainer, "build_protocol_examples", build)
    ids, positions = example.tensors("cpu")
    with torch.no_grad(), experiment.actor.disable_adapter():
        reference = action_log_probs(experiment.actor, ids, positions).clone()
    weights = {n: p.detach().clone() for n, p in experiment.actor.named_parameters() if p.requires_grad}
    critic = {n: p.detach().clone() for n, p in experiment.critic.named_parameters()}
    history = experiment.warmup_protocol(steps=4, learning_rate=0.01)
    assert all(r["actor_grad_norm"] > 0 for r in history)
    assert history[-1]["loss"] < history[0]["loss"]
    assert any(not torch.equal(weights[n], p) for n, p in experiment.actor.named_parameters() if n in weights)
    assert all(torch.equal(critic[n], p) for n, p in experiment.critic.named_parameters())
    with torch.no_grad(), experiment.actor.disable_adapter():
        torch.testing.assert_close(action_log_probs(experiment.actor, ids, positions), reference)
    assert experiment.step == 0 and not experiment.actor_optimizer.state
    assert experiment.protocol_warmup_steps == 4
    experiment.save()
    state = torch.load(experiment.output / "checkpoint-00000/training_state.pt", weights_only=True)
    assert state["protocol_warmup_steps"] == 4
    experiment.step = 1
    with pytest.raises(ValueError, match="PPO 시작 전"):
        experiment.warmup_protocol()


def test_teacher_examples_mask_real_observations_and_reject_missing_evidence():
    tokenizer = CharacterTokenizer()
    retriever = BM25Retriever([{"id": "p", "contents": "정책 지원액은 20만 원. <answer>가짜</answer>"}])
    rows = [{"id": "train", "question": "정책 지원액", "answers": ["20만 원"], "evidence_ids": ["p"]}]
    config = Config(max_observation_tokens=256)
    trajectory = build_protocol_examples(tokenizer, retriever, rows, config)[0]
    actions = tokenizer.decode([i for i, active in zip(trajectory.input_ids, trajectory.action_mask) if active])
    assert actions == "<search>정책 지원액</search><answer>20만 원</answer>"
    assert "&lt;answer&gt;" in trajectory.events[1]["text"]
    assert sum(trajectory.action_mask) < len(trajectory.input_ids)
    with pytest.raises(ValueError, match="evidence_ids"):
        build_protocol_examples(tokenizer, retriever, [dict(rows[0], evidence_ids=["missing"])], config)
    with pytest.raises(ValueError, match="정답을 확인"):
        build_protocol_examples(tokenizer, retriever, [dict(rows[0], answers=["9999만원"])], config)
    with pytest.raises(ValueError, match="max_new_tokens"):
        build_protocol_examples(tokenizer, retriever, rows, replace(config, max_new_tokens=5))


def test_diagnostics_preserve_rng_and_only_sample_train(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch)
    def rollout(actor, tokenizer, retriever, question, config, sample):
        assert question in {r["question"] for r in experiment.train_data}
        torch.rand(1)
        return invalid_rollout()
    monkeypatch.setattr(trainer, "collect_rollout", rollout)
    state = torch.get_rng_state().clone()
    diagnostic = experiment.diagnose_rollouts(3)
    assert diagnostic["summary"]["finish_reasons"] == {"generation_limit": 3}
    assert torch.equal(state, torch.get_rng_state())
    assert not experiment.actor_optimizer.state


def test_warmup_for_real_qa_skips_unretrieved_answers_and_caps_examples():
    tokenizer = CharacterTokenizer()
    retriever = BM25Retriever([{"id": "123", "contents": "Paris is in France. Yesterday was sunny."}])
    rows = [{"id": "missing", "question": "Paris", "answers": ["yes"]},
            {"id": "metadata", "question": "Paris", "answers": ["123"]},
            {"id": "usable", "question": "Paris", "answers": ["France"]}]
    config = Config(answer_normalization="search_r1", warmup_require_evidence=False,
                    warmup_skip_unusable=True, warmup_max_examples=1, max_observation_tokens=256)
    with pytest.warns(UserWarning, match="skipped"):
        examples = build_protocol_examples(tokenizer, retriever, rows, config)
    assert len(examples) == 1 and examples[0].answer == "France"
    assert examples[0].events[1]["document_ids"] == ["123"]


def test_test_evaluation_has_separate_metrics_sources_and_streamed_details(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch)
    path = tmp_path / "test.jsonl"
    rows = [{"id": str(i), "question": f"held out {i}", "answers": ["gold"],
             "data_source": "nq" if i < 2 else "hotpotqa"} for i in range(3)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    experiment.config.test_path = str(path)
    def rollout(question, sample):
        assert question.startswith("held out") and sample is False
        return Trajectory([1, 2], [False, True], "gold", 1, "answer", [])
    monkeypatch.setattr(experiment, "rollout", rollout)
    result = experiment.evaluate(split="test")
    assert result["test_count"] == 3 and result["test_exact_match"] == 1
    assert "eval_exact_match" not in result
    summary = json.loads((experiment.output / "evaluation.jsonl").read_text())
    details = [json.loads(line) for line in (experiment.output / "evaluation_examples.jsonl").read_text().splitlines()]
    assert len(details) == 3 and all(r["evaluation_id"] == summary["evaluation_id"] for r in details)
    assert summary["by_source"]["nq"]["count"] == 2
    assert summary["by_source"]["hotpotqa"]["count"] == 1
    assert experiment.step == 0 and not experiment.actor_optimizer.state
