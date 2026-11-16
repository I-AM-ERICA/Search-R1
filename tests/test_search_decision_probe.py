import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_search_decisions import prepare_inputs, ids_sha, summarize
from policy_r1.config import Config
from policy_r1.rollout import prompt_ids
from policy_r1.observations import observation_ids
from test_rollout import CharacterTokenizer


class PromptTokenizer(CharacterTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return self.encode("\n".join(m["content"] for m in messages), add_special_tokens=False)


def example():
    tokenizer = PromptTokenizer()
    config = Config(max_prompt_tokens=1200, max_seq_length=3000, max_new_tokens=100)
    question, prompt = "지원 조건?", "Use search."
    ids = prompt_ids(tokenizer, question, config, prompt)
    hits = [{"id": "d1", "contents": "가상 문서: 조건은 아직 확인되지 않았다."}]
    observation = observation_ids(tokenizer, hits, config.max_observation_tokens)
    search = "<search>지원 조건</search>"
    context = dict(arm="natural", case_id="c1", question=question, hits=hits, query="지원 조건",
                   search_text=search, search_tokens=len(tokenizer.encode(search, add_special_tokens=False)),
                   observation=tokenizer.decode(observation), observation_tokens=len(observation),
                   original_prompt_tokens=len(ids), original_prompt_sha256=ids_sha(ids), first_empty=False)
    return tokenizer, config, context, prompt


def test_translation_changes_instruction_but_preserves_same_observation_tokens():
    tokenizer, config, context, prompt = example()
    prepared = prepare_inputs(tokenizer, config, [context], prompt)
    after = [r for r in prepared if r["stage"] == "after_first_search"]
    tail = context["search_tokens"] + context["observation_tokens"]
    assert len(after) == 2
    assert after[0]["input_ids"][-tail:] == after[1]["input_ids"][-tail:]
    assert after[0]["input_ids"][:-tail] != after[1]["input_ids"][:-tail]
    assert all("answer" not in r for r in prepared)


def test_corrupt_source_evidence_is_rejected_instead_of_used_as_new_context():
    tokenizer, config, context, prompt = example()
    changed = copy.deepcopy(context)
    changed["hits"][0]["contents"] = "변조된 근거"
    with pytest.raises(ValueError, match="visible source evidence"):
        prepare_inputs(tokenizer, config, [changed], prompt)
    changed = copy.deepcopy(context)
    changed["original_prompt_sha256"] = "bad"
    with pytest.raises(ValueError, match="Tokenizer/prompt"):
        prepare_inputs(tokenizer, config, [changed], prompt)


def test_counterfactual_prompt_never_truncates_evidence_to_fit():
    tokenizer, config, context, prompt = example()
    config.max_seq_length = 300
    with pytest.raises(ValueError, match="No truncation"):
        prepare_inputs(tokenizer, config, [context], prompt)


def test_decision_summary_does_not_call_a_search_a_correct_answer():
    summary = summarize([dict(stage="after_first_search", arm="first-empty", language="ko",
                             variant="trained", action="search", query_changed=True, hit_token_limit=False)])
    result = next(iter(summary.values()))
    assert result["actions"] == {"search": 1}
    assert "accuracy" not in result and "success" not in result
