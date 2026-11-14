import io
import json
import socket
import ssl
from http.client import IncompleteRead
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

from policy_r1.config import Config
from policy_r1.inference import PolicyInference, InferenceFailure
from policy_r1.rollout import RolloutCancelled
from policy_r1.web_search import SerpAPIRetriever, SearchUnavailable
from test_rollout import CharacterTokenizer, ScriptedActor


def organic():
    return {"organic_results": [
        {"title": "정책 공고", "link": "https://example.go.kr/notice/1", "snippet": "정책 안내 요약"},
        {"title": "중복", "link": "https://example.go.kr/notice/1", "snippet": "중복"},
        {"title": "잘못된 URL", "link": "javascript:alert(1)", "snippet": "bad"},
        {"title": "다른 공고", "link": "https://example.go.kr/notice/2", "snippet": "공식 안내 확인"}]}


def test_serpapi_request_results_cache_and_provenance():
    requests = []
    def open_request(request, timeout):
        requests.append(request)
        assert timeout == 10
        args = parse_qs(urlsplit(request.full_url).query)
        assert args["engine"] == ["google"] and args["hl"] == ["ko"] and args["gl"] == ["kr"]
        assert args["api_key"] == ["dummy-test-key"]
        return io.BytesIO(json.dumps(organic()).encode())
    search = SerpAPIRetriever("dummy-test-key", opener=open_request)
    hits = search.search("지원 정책", 3, timeout=10)
    assert len(hits) == 2 and hits[0]["url"] == "https://example.go.kr/notice/1"
    assert "원문 미확인" in hits[0]["contents"]
    hits[0]["title"] = "caller mutation"
    assert search.search("지원 정책", 3)[0]["title"] == "정책 공고"
    assert len(requests) == 1


def test_naver_web_request_and_result_mapping_excludes_ads():
    requests = []
    def open_request(request, timeout):
        args = parse_qs(urlsplit(request.full_url).query)
        requests.append(args)
        assert args == {"engine": ["naver"], "query": ["지원 정책"],
                        "where": ["nexearch"], "api_key": ["dummy-test-key"]}
        payload = {"web_results": organic()["organic_results"],
                   "ads_results": [{"title": "광고", "link": "https://ads.example.com", "snippet": "광고"}],
                   "search_parameters": {"engine": "naver", "query": "지원 정책"}}
        return io.BytesIO(json.dumps(payload).encode())
    search = SerpAPIRetriever("dummy-test-key", engine="naver", opener=open_request)
    hits = search.search("지원 정책", 3)
    assert len(hits) == 2 and hits[0]["url"] == "https://example.go.kr/notice/1"
    assert search.last_diagnostics["query_matches_response"] is True
    assert search.last_diagnostics["result_field"] == "web_results"
    assert search.search("지원 정책", 3) == hits and len(requests) == 1
    with pytest.raises(ValueError, match="engine"):
        SerpAPIRetriever("dummy", engine="unsupported")


def test_naver_missing_or_invalid_web_results_do_not_use_google_or_ads():
    payload = organic()
    search = SerpAPIRetriever("dummy", engine="naver", cache_size=0,
                             opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    assert search.search("query") == []
    payload["web_results"] = {}
    with pytest.raises(SearchUnavailable, match="invalid results structure"):
        search.search("query")


@pytest.mark.parametrize("error", [
    HTTPError("https://serpapi.com/?api_key=secret-value", 401, "secret-value", {}, None),
    URLError("secret-value"), TimeoutError("secret-value")])
def test_search_errors_do_not_leak_credentials(error):
    def fail(*args, **kwargs):
        raise error
    with pytest.raises(SearchUnavailable) as result:
        SerpAPIRetriever("secret-value", opener=fail).search("query")
    assert "secret-value" not in str(result.value)


@pytest.mark.parametrize("failure,code", [
    (TimeoutError("secret-value"), "TIMEOUT"),
    (URLError(TimeoutError("secret-value")), "TIMEOUT"),
    (URLError(socket.gaierror(-2, "secret-value")), "DNS_ERROR"),
    (URLError(ssl.SSLCertVerificationError(1, "secret-value")), "TLS_CERTIFICATE_ERROR"),
    (ssl.SSLError(1, "secret-value"), "TLS_ERROR"),
    (ConnectionResetError("secret-value"), "CONNECTION_ERROR"),
    (IncompleteRead(b"secret-value"), "HTTP_PROTOCOL_ERROR"),
    (URLError("secret-value"), "NETWORK_ERROR"),
])
def test_diagnostic_failure_codes_and_timeout(failure, code):
    def fail(request, timeout):
        assert timeout == 45
        raise failure
    with pytest.raises(SearchUnavailable) as result:
        SerpAPIRetriever("secret-value", timeout=45, opener=fail).search("query")
    assert code in str(result.value) and "secret-value" not in str(result.value)
    assert "socket timeout=45s" in str(result.value)


def test_invalid_json_is_distinct_from_transport_error():
    with pytest.raises(SearchUnavailable, match="INVALID_JSON") as result:
        SerpAPIRetriever("secret-value", opener=lambda *a, **kw: io.BytesIO(
            b'<html>secret-value</html>')).search("query")
    assert "secret-value" not in str(result.value)


def test_provider_error_and_missing_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SERPAPI_API_KEY"):
        SerpAPIRetriever()
    with pytest.raises(SearchUnavailable) as error:
        SerpAPIRetriever("secret-value", opener=lambda *a, **kw: io.BytesIO(
            b'{"error":"secret-value"}')).search("query")
    assert "secret-value" not in str(error.value)


def test_explicit_empty_success_is_not_an_api_failure():
    payload = {"search_metadata": {"status": "Success", "id": "a" * 24},
               "error": "Naver hasn't returned any results for this query."}
    search = SerpAPIRetriever("dummy", engine="naver", cache_size=0,
        opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    assert search.search("query") == []
    assert search.last_diagnostics["provider_empty_result"] is True
    payload["search_metadata"]["status"] = "Error"
    with pytest.raises(SearchUnavailable) as failed:
        search.search("query")
    assert failed.value.code == "PROVIDER_ERROR" and failed.value.search_id == "a" * 24
    payload["search_metadata"]["status"] = "Success"
    payload["error"] = "secret-value quota error"
    with pytest.raises(SearchUnavailable) as failed:
        search.search("query")
    assert failed.value.code == "PROVIDER_ERROR" and "secret-value" not in str(failed.value)


def test_placeholder_results_are_filtered_before_top_k_without_guessing_urls():
    bad = [
        {"title": "https://official.go.kr/truncated...", "link": "https://www.google.com/goto?url=opaque",
         "snippet": "이 페이지에 관한 정보가 없습니다."},
        {"title": "공고", "link": "https://example.go.kr/empty", "snippet": "이 페이지에 관한 정보가 없습니다. 자세히 알아보기"},
        {"title": "Notice", "link": "https://example.go.kr/empty-en", "snippet": "No information is available for this page. Learn why"},
        {"title": "공고", "link": "https://www.google.co.kr/url?q=https://example.go.kr", "snippet": "지원사업 공고 안내"},
    ]
    payload = {"organic_results": bad + organic()["organic_results"], "search_parameters": {"q": "정책"}}
    retriever = SerpAPIRetriever("dummy", opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    hits = retriever.search("정책", 1)
    assert len(hits) == 1 and hits[0]["url"] == "https://example.go.kr/notice/1"
    assert retriever.last_diagnostics["query_matches_response"] is True
    assert retriever.last_diagnostics["rejected"]["google_redirect"] == 2
    assert retriever.last_diagnostics["rejected"]["missing_description"] == 2
    assert retriever.last_diagnostics["usable_results"] == 2
    payload["organic_results"] = bad
    service = engine(["<search>정책</search>", "<answer>지원금을 받으세요.</answer>"], payload)
    with pytest.raises(InferenceFailure, match="No retrieved evidence"):
        service.answer("정책 질문")


def engine(responses, payload=None, max_prompt=1800):
    tokenizer = CharacterTokenizer()
    actor = ScriptedActor(responses, tokenizer)
    search = SerpAPIRetriever("dummy", opener=lambda *args, **kwargs: io.BytesIO(
        json.dumps(payload if payload is not None else organic()).encode()))
    config = Config(max_seq_length=6000, max_prompt_tokens=max_prompt, max_observation_tokens=1500,
                    max_new_tokens=192, max_searches=2)
    return PolicyInference(tokenizer, actor, config, search)


def test_actual_search_rollout_returns_only_real_sources():
    service = engine(["<search>지원 정책</search>", "<answer>공식 공고에서 신청 조건을 확인해 주세요.</answer>"])
    result = service.answer("음식점 지원정책을 찾아줘", region="서울", category="음식점")
    assert result["searches"] == 1 and len(result["sources"]) == 2
    assert result["sources"][0]["url"] == "https://example.go.kr/notice/1"
    assert result["evidence_mode"] == "search_snippets"


def test_model_can_search_again_after_explicit_empty_success():
    service = engine(["<search>첫 검색</search>", "<search>수정 검색</search>", "<answer>공고를 확인해 주세요.</answer>"])
    payloads = iter([
        {"search_metadata": {"status": "Success"}, "error": "Google hasn't returned any results for this query."},
        organic(),
    ])
    service.retriever._open = lambda *a, **kw: io.BytesIO(json.dumps(next(payloads)).encode())
    result = service.answer("정책을 찾아줘")
    assert result["searches"] == 2 and len(result["sources"]) == 2


def test_official_excerpt_reaches_rollout_with_correct_evidence_label():
    payload = {"organic_results": [{"title": "공식 공고", "link": "https://www.semas.or.kr/notice",
                                   "snippet": None}]}
    service = engine(["<search>지원 정책</search>", "<answer>공식 공고의 조건을 확인해 주세요.</answer>"], payload)
    service.retriever._fetch_page = lambda *a, **kw: "지원 대상 및 접수 기간 안내 " * 12
    result = service.answer("지원정책을 찾아줘")
    assert result["evidence_mode"] == "search_snippets_and_page_excerpts"
    assert result["sources"][0]["url"] == "https://www.semas.or.kr/notice"


@pytest.mark.parametrize("responses,payload", [
    (["<answer>지원금은 999만원입니다.</answer>"], None),
    (["<search>정책</search>", "<answer>999만원</answer>"], {"organic_results": []}),
    (["잘못된 형식"], None)])
def test_unsupported_or_invalid_answers_are_not_served(responses, payload):
    with pytest.raises(InferenceFailure):
        engine(responses, payload).answer("정책 질문")


def test_cancel_before_generation_and_keep_latest_question_when_history_is_long():
    service = engine([])
    with pytest.raises(RolloutCancelled):
        service.answer("정책 질문", cancelled=lambda: True)
    latest = "지금 신청 가능한 지원이 있나요?"
    text = service._question("다음은 같은 사용자의 이전 대화입니다.\n" + "오래된 대화 " * 2000 +
                             "\n현재 사용자 질문: " + latest, "서울", "음식점")
    assert text.endswith(latest)
    assert len(text) < 3000


def test_actor_only_loader_restores_saved_lora(tmp_path, monkeypatch):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from policy_r1.models import prepare_backbone
    from policy_r1.inference import load_actor
    config = Config(load_in_4bit=False, gradient_checkpointing=False)
    tiny = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    original = prepare_backbone(LlamaForCausalLM(tiny), config, "CAUSAL_LM")
    with pytest.warns(UserWarning):
        original.save_pretrained(tmp_path / "actor")
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    (tmp_path / "metadata.json").write_text('{"resolved_model_revision":"fixed-revision"}')
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    def base(*args, **kwargs):
        assert kwargs["revision"] == "fixed-revision"
        return LlamaForCausalLM(tiny)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", base)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: SimpleNamespace(pad_token_id=0))
    _, restored, _ = load_actor(tmp_path)
    expected = dict(original.named_parameters())
    for name, value in restored.named_parameters():
        if "lora_" in name:
            torch.testing.assert_close(value, expected[name])
        assert value.requires_grad is False
