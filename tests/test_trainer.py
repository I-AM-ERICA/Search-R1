# 역할: 인터넷/GPU 없이 작은 Llama로 전체 PPO 업데이트와 저장·재개가 연결되는지 검증합니다.
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from policy_r1.config import Config
from policy_r1.models import action_log_probs
from policy_r1.rollout import Trajectory
from policy_r1 import trainer
from test_ppo import tiny_models


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 5]

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / "test_tokenizer.json").write_text("{}")


def test_real_ppo_update_and_checkpoint_resume(tmp_path, monkeypatch):
    def factory(config, actor_adapter=None, critic_adapter=None):
        actor, critic = tiny_models(config)
        for model, path in [(actor, actor_adapter), (critic.backbone, critic_adapter)]:
            if path:
                set_peft_model_state_dict(model, load_file(str(path / "adapter_model.safetensors")))
        return TinyTokenizer(), actor, critic

    answers = {}
    for filename in ["data/demo/train.jsonl", "data/demo/val.jsonl"]:
        for line in Path(filename).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            answers[row["question"]] = row["answers"][0]

    def rollout(actor, tokenizer, retriever, question, config, sample):
        return Trajectory([1, 5, 6, 7, 8, 9, 10],
                          [False, False, True, True, False, True, True],
                          answers[question], 1, "answer", [])

    monkeypatch.setattr(trainer, "load_models", factory)
    monkeypatch.setattr(trainer, "collect_rollout", rollout)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test double")
    for name in ("train", "val", "corpus"):
        (tmp_path / f"{name}.jsonl").write_bytes(Path(f"data/demo/{name}.jsonl").read_bytes())
    config = Config(load_in_4bit=False, total_updates=2, rollout_batch_size=2, train_sampling="shuffle", eval_max_samples=1,
                    train_path=str(tmp_path / "train.jsonl"), val_path=str(tmp_path / "val.jsonl"),
                    corpus_path=str(tmp_path / "corpus.jsonl"),
                    eval_every=2, save_every=2, output_dir=str(tmp_path / "run"))
    with pytest.warns(UserWarning):
        experiment = trainer.PPOExperiment(config)
    original = {n: p.detach().clone() for n, p in experiment.actor.named_parameters() if p.requires_grad}
    history = experiment.train()
    assert len(history) == 2 and history[-1]["eval_exact_match"] == 1
    assert history[-1]["eval_count"] == 1
    assert history[-1]["train_examples_seen"] == len(experiment.train_data)
    traces = [json.loads(line) for line in (experiment.output / "rollouts.jsonl").read_text().splitlines()]
    assert len({r["id"] for r in traces}) == len(experiment.train_data)
    assert experiment.evaluate(full=True)["eval_count"] == len(experiment.val_data)
    assert any(not torch.equal(original[n], p) for n, p in experiment.actor.named_parameters() if n in original)
    assert experiment.critic.value_head.weight.abs().sum() > 0
    checkpoint = tmp_path / "run/checkpoint-00002"
    assert (checkpoint / "training_state.pt").exists()
    ids, positions = rollout(None, None, None, next(iter(answers)), config, False).tensors("cpu")
    experiment.actor.eval()
    with torch.no_grad():
        expected = action_log_probs(experiment.actor, ids, positions)
    with pytest.warns(UserWarning):
        resumed = trainer.PPOExperiment(replace(config, total_updates=3, output_dir=str(tmp_path / "resumed")), checkpoint)
    resumed.actor.eval()
    with torch.no_grad():
        torch.testing.assert_close(action_log_probs(resumed.actor, ids, positions), expected)
    assert resumed.step == 2
    assert resumed.train_order == experiment.train_order
    assert resumed.train_examples_seen == experiment.train_examples_seen
    assert len(resumed.actor_optimizer.state) > 0
    torch.testing.assert_close(resumed.critic.value_head.weight, experiment.critic.value_head.weight)
    assert resumed.train()[0]["step"] == 3
    assert (tmp_path / "resumed/checkpoint-00003").exists()
    changed = Path(config.train_path).read_text().splitlines()
    Path(config.train_path).write_text("\n".join(reversed(changed)) + "\n")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="데이터 내용/순서"):
        trainer.PPOExperiment(replace(config, output_dir=str(tmp_path / "changed")), checkpoint)
