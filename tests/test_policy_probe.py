import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from test_web_serving import engine, organic


spec = importlib.util.spec_from_file_location("policy_probe", Path(__file__).resolve().parents[1] / "scripts/diagnose_policy.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_probe_distinguishes_direct_answer_from_empty_search():
    direct = engine(["<answer>공고를 확인하세요.</answer>"])
    previous = direct.retriever
    report = module.probe(direct, "정책 질문")
    assert report["status"] == "failed" and report["trajectory"]["searches"] == 0
    assert report["search_calls"] == [] and direct.retriever is previous
    empty = engine(["<search>검색어</search>", "<answer>공고를 확인하세요.</answer>"], {"organic_results": []})
    report = module.probe(empty, "정책 질문")
    assert report["status"] == "failed" and report["trajectory"]["searches"] == 1
    assert report["search_calls"][0]["returned_hits"] == []


def test_probe_confirms_sources_reached_observation_without_answer_text():
    service = engine(["<search>검색어</search>", "<answer>답변본문은추적에서제외</answer>"])
    report = module.probe(service, "정책 질문")
    assert report["status"] == "success" and report["source_count"] == 2
    observation = next(e for e in report["trajectory"]["events"] if e["type"] == "observation")
    assert len(observation["visible_ids"]) == 2
    assert "답변본문은추적에서제외" not in str(report)


@pytest.mark.parametrize("separator", [" ", " / "])
def test_reported_context_echo_recovers_through_cleaned_and_broader_queries(separator):
    question = "소상공인시장진흥공단 정책자금의 공통 지원 자격을 공식 자료로 확인해줘."
    service = engine([])
    prefix = separator.join(service._question(question, "서울", "음식점").splitlines()[:3])
    # Reproduce the user's generated search action, then a final answer.
    from test_rollout import ScriptedActor
    service.actor = ScriptedActor([f"<search>{prefix}{separator}{question}</search>", "<answer>공고를 확인해 주세요.</answer>"], service.tokenizer)
    queries = []
    def response(request, **kwargs):
        query = parse_qs(urlsplit(request.full_url).query)["q"][0]
        queries.append(query)
        payload = {"organic_results": []} if len(queries) == 1 else organic()
        if len(queries) > 1:
            for row in payload["organic_results"]:
                row["link"] = row["link"].replace("example.go.kr", "www.semas.or.kr")
        return io.BytesIO(json.dumps(payload).encode())
    service.retriever._open = response
    report = module.probe(service, question)
    assert report["status"] == "success" and report["search_requests"] == 2
    assert [call["query"] for call in report["search_calls"]] == queries == [
        "소상공인시장진흥공단 정책자금 공통 지원 자격", "소상공인시장진흥공단 정책자금 지원"]
    assert report["trajectory"]["searches"] == 1 and report["source_count"] == 2


def test_official_answer_uses_only_verified_sources_after_broadening():
    service = engine(["<search>소상공인시장진흥공단 정책자금 공통 지원 자격</search>",
                      "<answer>공식 공고를 확인하세요.</answer>"])
    payloads = iter([
        {"organic_results": [{"title": "공단 공식 안내", "link": "https://private.example/", "snippet": "공식인 것처럼 표시"}]},
        {"organic_results": [{"title": "소상공인시장진흥공단", "link": "https://www.semas.or.kr/notice", "snippet": "지원 안내"}]},
    ])
    service.retriever._open = lambda *a, **kw: io.BytesIO(json.dumps(next(payloads)).encode())
    report = module.probe(service, "공통 지원 자격을 공식 자료로 확인해줘", show_answer=True)
    assert report["status"] == "success" and report["source_count"] == 1
    assert report["answer"] == "공식 공고를 확인하세요."
    assert report["sources"][0]["url"] == "https://www.semas.or.kr/notice"
    observed = next(e for e in report["trajectory"]["events"] if e["type"] == "observation")
    assert len(observed["returned_ids"]) == 1 and "private.example" not in observed["preview"]
    assert report["source_selection"][0]["accepted_count"] == 0


def test_nonofficial_only_results_do_not_satisfy_an_official_request():
    service = engine(["<search>공통 지원 자격</search>", "<answer>단정적인 답변</answer>"],
                     {"organic_results": [{"title": "공식 자료", "link": "https://private.example", "snippet": "안내"}]})
    report = module.probe(service, "공식 자료로 확인해줘", show_answer=True)
    assert report["status"] == "failed" and "answer" not in report
    assert all(s["accepted_count"] == 0 for s in report["source_selection"])


def test_generation_override_recovers_truncated_answer_without_modifying_original_config():
    from policy_r1.inference import with_generation_limit
    response = "<answer>" + "공식 공고의 조건을 확인하세요. " * 12 + "</answer>"
    assert 192 < len(response) < 512
    service = engine(["<search>지원 정책</search>", response])
    original = service.config
    report = module.probe(service, "지원 정책", show_invalid_output=True)
    assert report["status"] == "failed" and report["trajectory"]["finish_reason"] == "generation_limit"
    invalid = report["trajectory"]["events"][-1]
    assert invalid["starts_with_answer"] and not invalid["has_answer_close"]
    assert invalid["output_head"].startswith("<answer>")
    longer = engine(["<search>지원 정책</search>", response])
    longer.config = with_generation_limit(original, 512)
    report = module.probe(longer, "지원 정책", show_answer=True)
    assert report["status"] == "success" and report["answer"] == response[8:-9].strip()
    assert original.max_new_tokens == 192 and longer.config.max_new_tokens == 512


def test_invalid_output_is_optional_and_larger_limit_does_not_accept_bad_format():
    from policy_r1.inference import with_generation_limit
    service = engine(["무효출력표식 " * 100])
    service.config = with_generation_limit(service.config, 512)
    report = module.probe(service, "정책 질문")
    assert report["status"] == "failed"
    assert "무효출력표식" not in str(report)
    assert "output_head" not in report["trajectory"]["events"][-1]


def test_invalid_generation_limits_are_rejected():
    import pytest
    from policy_r1.inference import with_generation_limit
    config = engine([]).config
    for bad in (0, -1, 1025, 1.5, True):
        with pytest.raises(ValueError):
            with_generation_limit(config, bad)
    assert with_generation_limit(config) is config


def test_focused_section_is_the_actual_observation_and_returned_source():
    from test_evidence_focus import sample
    payload = {"organic_results": [{"title": "공식 안내", "link": "https://www.semas.or.kr/notice", "snippet": None}]}
    service = engine(["<search>정책자금 공통 지원 자격</search>", "<answer>공통 조건과 제외 조건을 확인하세요.</answer>"], payload)
    service.retriever._fetch_page = lambda *a, **kw: sample()
    report = module.probe(service, "공통 지원 자격을 공식 자료로 확인해줘", show_answer=True)
    assert report["status"] == "success"
    focus = report["evidence_focus"][0]
    assert focus["selected_chars"] < focus["original_chars"]
    assert focus["section_heading"] == "공통 지원 자격"
    observation = next(e for e in report["trajectory"]["events"] if e["type"] == "observation")
    assert "금리 안내" not in observation["preview"]
