from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from policy_r1.config import Config
from policy_r1.models import prepare_backbone
from test_rollout import CharacterTokenizer, ScriptedActor

spec = importlib.util.spec_from_file_location("summary_compare", Path(__file__).resolve().parents[1] / "scripts/compare_policy_summary.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_disabling_lora_matches_base_logits_and_restores_adapter_without_changing_weights():
    torch.manual_seed(7)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)).eval()
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        expected = base(ids).logits.clone()
    actor = prepare_backbone(base, Config(load_in_4bit=False, gradient_checkpointing=False, lora_rank=2), "CAUSAL_LM").eval()
    with torch.no_grad():
        for name, param in actor.named_parameters():
            if "lora_B" in name:
                param.normal_(0, .2)
    original = {name: param.detach().clone() for name, param in actor.named_parameters()}
    with torch.no_grad(), comparison.adapter_variant(actor, "base"):
        assert torch.allclose(actor(ids).logits, expected, atol=1e-6)
    with torch.no_grad(), comparison.adapter_variant(actor, "trained"):
        assert not torch.allclose(actor(ids).logits, expected, atol=1e-6)
    with pytest.raises(RuntimeError, match="simulated"):
        with comparison.adapter_variant(actor, "base"):
            raise RuntimeError("simulated")
    assert actor.get_model_status().enabled is True
    assert all(torch.equal(param, original[name]) for name, param in actor.named_parameters())


def test_inputs_preserve_all_source_text_and_refuse_silent_truncation():
    dataset, documents = comparison.read_dataset("evaluations/policy_summary/cases.json")
    tokenizer = CharacterTokenizer()
    case = dataset["cases"][0]
    document = documents[case["document_id"]]
    messages, ids = comparison.build_inputs(tokenizer, case, document, "plain", 10000)
    assert document["text"] in messages[-1]["content"]
    assert document["source_url"] in messages[-1]["content"]
    assert "review_points" not in messages[-1]["content"]
    with pytest.raises(ValueError, match="no text was truncated"):
        comparison.build_inputs(tokenizer, case, document, "plain", len(ids)-1)


class ToggleActor(ScriptedActor):
    enabled = True

    def get_model_status(self):
        return SimpleNamespace(enabled=self.enabled)

    @contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


def test_same_inputs_are_used_for_each_pair_and_quality_is_not_automatically_graded(tmp_path, monkeypatch):
    dataset, _ = comparison.read_dataset("evaluations/policy_summary/cases.json")
    calls = []
    def generate(actor, tokenizer, ids, variant, mode, max_new_tokens, seed):
        calls.append((list(ids), variant, mode))
        return {"variant": variant, "mode": mode, "output": "답변", "answer": "답변", "format_valid": True,
                "hit_token_limit": False, "completed": True, "generated_tokens": 2, "seconds": .01}
    monkeypatch.setattr(comparison, "generate_one", generate)
    output = tmp_path / "report"
    results = comparison.run_comparison(None, CharacterTokenizer(), Config(max_seq_length=6000), dataset, output)
    assert len(results) == 24
    for index in range(0, len(calls), 2):
        assert calls[index][0] == calls[index+1][0]
        assert {calls[index][1], calls[index+1][1]} == {"base", "trained"}
    assert all(r["human_review"]["correctness_0_to_2"] is None for r in results)
    assert len((output / "results.jsonl").read_text().splitlines()) == 24
    assert (output / "report.txt").is_file()
    with pytest.raises(FileExistsError):
        comparison.run_comparison(None, CharacterTokenizer(), Config(max_seq_length=6000), dataset, output)


def test_invalid_and_truncated_answers_are_recorded_instead_of_dropped():
    tokenizer = CharacterTokenizer()
    actor = ToggleActor(["<search>검색</search>", "x" * 50], tokenizer)
    first = comparison.generate_one(actor, tokenizer, [1, 5], "base", "tagged", 64, 42)
    assert first["output"] == "<search>검색</search>" and not first["completed"]
    second = comparison.generate_one(actor, tokenizer, [1, 5], "trained", "plain", 10, 42)
    assert second["hit_token_limit"] and not second["completed"]
    assert actor.enabled


def test_optional_raw_tokens_do_not_change_generation():
    tokenizer = CharacterTokenizer()
    text = '<answer>done</answer>'
    actor = ToggleActor([text, text], tokenizer)
    plain = comparison.generate_one(actor, tokenizer, [1,5], 'trained', 'tagged', 64, 42)
    detail = comparison.generate_one(actor, tokenizer, [1,5], 'trained', 'tagged', 64, 42,
                                     include_generation_details=True)
    assert detail['output'] == plain['output'] == text
    assert len(detail['generated_token_ids']) == detail['generated_tokens']
    assert detail['raw_output'] == tokenizer.decode(detail['generated_token_ids'], skip_special_tokens=False)
    assert 'raw_output' not in plain


def test_source_edits_without_checksum_update_are_rejected(tmp_path):
    data = json.loads(Path("evaluations/policy_summary/cases.json").read_text())
    data["documents"][0]["text"] += " changed"
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        comparison.read_dataset(path)
