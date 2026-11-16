import copy
import json
import sys

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from policy_r1.sft_pilot import execution_plan, probe_inputs, probe_summary


def inputs():
    status = dict(ready_for_training=False, checks=dict(train_count=False, val_count=False,
                  reviewed_recovery=False, single_search=True, insufficient_evidence=True))
    splits = dict(train=[{}] * 48, val=[{}] * 15, test=[{}])
    settings = dict(proposed_sft=dict(epochs=1, gradient_accumulation=8, learning_rate=2e-5,
                                    max_grad_norm=1.0))
    return status, splits, settings


def test_small_data_requires_explicit_pilot_and_preserves_standard_status():
    status, splits, settings = inputs()
    originals = copy.deepcopy((status, splits, settings))
    with pytest.raises(ValueError, match='Data gates'):
        execution_plan(status, splits, settings)
    result = execution_plan(status, splits, settings, pilot=True, epochs=3)
    assert result['planned_updates'] == 18
    assert not result['standard_ready_for_training']
    assert (status, splits, settings) == originals
    with pytest.raises(ValueError, match='requires'):
        execution_plan(status, splits, settings, epochs=3)


def test_pilot_boundaries_and_missing_validation():
    status, splits, settings = inputs()
    for epochs in (0, 4):
        with pytest.raises(ValueError, match='1 to 3'):
            execution_plan(status, splits, settings, pilot=True, epochs=epochs)
    splits['train'] = [{}] * 81
    with pytest.raises(ValueError, match='30 optimizer'):
        execution_plan(status, splits, settings, pilot=True, epochs=3)
    splits['val'] = []
    with pytest.raises(ValueError, match='nonempty'):
        execution_plan(status, splits, settings, pilot=True)


def test_learning_rate_arm_is_explicit_bounded_and_does_not_mutate_source():
    status, splits, settings = inputs()
    result = execution_plan(status, splits, settings, pilot=True, epochs=3, learning_rate=1e-4)
    assert result['training_spec']['learning_rate'] == 1e-4
    assert result['planned_updates'] == 18
    assert settings['proposed_sft']['learning_rate'] == 2e-5
    with pytest.raises(ValueError, match='requires'):
        execution_plan(status, splits, settings, learning_rate=1e-4)
    for rate in (0, -1e-5, 2e-4, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite'):
            execution_plan(status, splits, settings, pilot=True, learning_rate=rate)


def test_probe_never_includes_the_target_action_or_future_observation():
    row = dict(id='v', split='val', input_ids=list(range(14)), spans=[
        dict(type='search', start=2, end=4), dict(type='observation', start=4, end=6),
        dict(type='search', start=6, end=8), dict(type='observation', start=8, end=10),
        dict(type='answer', start=10, end=14)])
    cases = probe_inputs([row])
    assert [c['input_ids'] for c in cases] == [list(range(2)), list(range(6)), list(range(10))]
    assert [c['expected_action'] for c in cases] == ['search', 'search', 'answer']
    with pytest.raises(ValueError, match='Only validation'):
        probe_inputs([dict(row, split='test')])
    summary = probe_summary([dict(stage='initial', action='invalid', expected_action='search', hit_token_limit=True)])
    assert summary['initial']['teacher_action_matches'] == 0
    assert summary['initial']['token_limit'] == 1


def test_pilot_training_saves_epochs_and_uses_only_training_rows(tmp_path, monkeypatch):
    # Real tiny CPU LoRA/optimizer; mock only hardware/loading and generation.
    # Integrity is tested separately in test_sft_training.py.
    from scripts import train_policy_sft as trainer
    from scripts import compare_policy_summary
    from policy_r1.config import Config
    from policy_r1.policy_sft import fingerprint
    from policy_r1.inference import POLICY_SYSTEM_PROMPT
    status, _, settings = inputs()
    row = dict(id='train', split='train', input_ids=[1, 2, 3, 4, 5, 6],
               action_mask=[False, False, True, False, True, True], spans=[
                   dict(type='search', start=2, end=3), dict(type='observation', start=3, end=4),
                   dict(type='answer', start=4, end=6)])
    val = dict(row, id='val', split='val', input_ids=[7, 8, 9, 10, 11, 12])
    splits = dict(train=[row], val=[val], test=[])
    settings.update(initialization='fresh_base_lora; do not resume checkpoint-00150',
                    model_environment=Config(load_in_4bit=False, lora_rank=2, max_seq_length=64,
                    max_prompt_tokens=16, max_new_tokens=16, max_observation_tokens=16,
                    model_revision='a'*40).to_dict())
    monkeypatch.setattr(trainer, 'inspect_bundle', lambda *a: (status, splits, settings, dict(prompt_sha256=fingerprint(POLICY_SYSTEM_PROMPT))))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: False)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                           max_position_embeddings=64))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: base)
    class Tokenizer:
        pad_token_id = 0
        def save_pretrained(self, path):
            path.mkdir(); (path/'fixture.json').write_text('{}')
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    def generate(*args):
        return dict(output='<search>test</search>', hit_token_limit=False)
    monkeypatch.setattr(compare_policy_summary, 'generate_one', generate)
    calls = []
    from policy_r1 import models
    original = models.action_log_probs
    def recording(actor, ids, positions):
        calls.append((torch.is_grad_enabled(), ids.tolist()[0]))
        return original(actor, ids, positions)
    monkeypatch.setattr(models, 'action_log_probs', recording)
    (tmp_path/'report.json').write_text('{}')
    output = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['train', '--data', str(tmp_path), '--tokenizer', str(tmp_path),
                        '--output-dir', str(output), '--pilot-small-data', '--epochs', '3',
                        '--learning-rate', '1e-4', '--execute'])
    trainer.main()
    assert [ids for grad, ids in calls if grad] == [row['input_ids']] * 3
    assert [ids for grad, ids in calls if not grad] == [val['input_ids']] * 4
    metrics = [json.loads(s) for s in (output/'validation.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in metrics] == [0, 1, 2, 3]
    assert all((output/f'checkpoint-{i:05d}/actor/adapter_model.safetensors').exists() for i in (1, 2, 3))
    assert (output/'TRAINING_COMPLETE.json').exists()
    assert not (output/'INCOMPLETE.txt').exists()
    optimizer = torch.load(output/'checkpoint-00003/optimizer.pt', weights_only=True)
    assert optimizer['optimizer']['param_groups'][0]['lr'] == 1e-4
