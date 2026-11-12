# 역할: GAE·PPO clipping 수식, action log-prob 정렬, 실제 LoRA/Value 역전파를 작은 모델로 검증합니다.
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel

from policy_r1.config import Config
from policy_r1.models import ValueModel, action_log_probs, prepare_backbone
from policy_r1.objectives import compute_gae, policy_loss, value_loss


def tiny_models(config=None):
    config = config or Config(load_in_4bit=False)
    settings = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=4,
                           num_key_value_heads=2, max_position_embeddings=2048,
                           attention_dropout=0.0)
    actor = prepare_backbone(LlamaForCausalLM(settings), config, "CAUSAL_LM")
    critic = ValueModel(prepare_backbone(LlamaModel(settings), config, "FEATURE_EXTRACTION"))
    return actor, critic


def test_hand_calculated_gae():
    advantage, returns = compute_gae(torch.tensor([0.0, 1.0]), torch.tensor([0.2, 0.3]), 1.0, 1.0)
    torch.testing.assert_close(advantage, torch.tensor([0.8, 0.7]))
    torch.testing.assert_close(returns, torch.tensor([1.0, 1.0]))


def test_signed_policy_clipping_and_value_clipping():
    ratio = torch.tensor([1.5, 0.5])
    losses, _ = policy_loss(ratio.log(), torch.zeros(2), torch.tensor([1.0, -1.0]), 0.2)
    torch.testing.assert_close(losses, torch.tensor([-1.2, 0.8]))
    values = value_loss(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0]), 0.2)
    torch.testing.assert_close(values, torch.tensor([0.32]))


def test_token_alignment_gradients_and_frozen_reference():
    torch.manual_seed(0)
    actor, critic = tiny_models()
    ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10]])
    # indices 3/4 are environment observations, indices 0/1 are prompt.
    positions = torch.tensor([2, 5, 6])
    actor.eval()
    with torch.no_grad():
        expected = actor(ids).logits[0, positions - 1].log_softmax(-1).gather(1, ids[0, positions, None]).squeeze(1)
        selected = action_log_probs(actor, ids, positions)
        torch.testing.assert_close(selected, expected)
        with actor.disable_adapter():
            reference_before = action_log_probs(actor, ids, positions).clone()
    actor.train()
    torch.testing.assert_close(action_log_probs(actor, ids, positions), selected)
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=0.01)
    (-action_log_probs(actor, ids, positions).mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in actor.named_parameters() if "lora_" in n)
    assert all(p.grad is None for n, p in actor.named_parameters() if "lora_" not in n)
    optimizer.step()
    actor.eval()
    with torch.no_grad(), actor.disable_adapter():
        torch.testing.assert_close(action_log_probs(actor, ids, positions), reference_before)
    critic.train()
    predictions = critic(ids, positions)
    assert predictions.shape == (3,)
    (predictions - 1).square().mean().backward()
    assert critic.value_head.weight.grad.abs().sum() > 0
    assert actor.get_base_model().model.embed_tokens.weight.data_ptr() != critic.backbone.get_base_model().embed_tokens.weight.data_ptr()


def test_generation_cached_log_probs_match_full_sequence():
    from transformers import GenerationConfig
    torch.manual_seed(3)
    actor, _ = tiny_models()
    actor.eval()
    with torch.no_grad():
        output = actor.generate(input_ids=torch.tensor([[1, 5, 6]]),
                                generation_config=GenerationConfig(
                                    do_sample=True, top_k=0, top_p=1.0, temperature=1.0,
                                    max_new_tokens=3, pad_token_id=0, use_cache=True),
                                return_dict_in_generate=True, output_scores=True)
        positions = torch.arange(3, output.sequences.shape[1])
        rollout_logp = torch.stack([scores[0].log_softmax(-1)[token]
                                   for scores, token in zip(output.scores, output.sequences[0, 3:])])
        torch.testing.assert_close(action_log_probs(actor, output.sequences, positions), rollout_logp,
                                   atol=1e-5, rtol=1e-5)
