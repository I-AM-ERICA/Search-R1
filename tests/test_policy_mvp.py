from types import SimpleNamespace
import time

import pytest

from policy_r1.config import Config
from policy_r1.mvp import MVPPolicyInference, validate_claims, search_plan, region_matches, source_identity
from policy_r1.rollout import RolloutCancelled


BODY = "지원 대상: 서울 소재 소상공인\n\n제외 조건: 휴업 또는 폐업 사업자 제외\n\n신청 기간: 2026년 11월 1일~11월 30일\n\n신청 방법: 소상공인24 온라인 신청"


def hit(url="https://www.bizinfo.go.kr/example"):
    return dict(id="web_test", title="테스트 공고 (실제 정책 아님)", url=url,
                snippet=BODY, contents=BODY, score=1., evidence_kind="official_page_excerpt")


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1]

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class Search:
    require_page_evidence = True

    def __init__(self, results):
        self.results, self.calls = iter(results), []

    def search(self, query, top_k, **kwargs):
        self.calls.append(query)
        return next(self.results)


def engine(search):
    return MVPPolicyInference(Tokenizer(), None, Config(max_seq_length=4096, max_observation_tokens=1400), search)


def test_rejects_invented_amount_source_and_field():
    import json
    good = dict(source_id="web_test", field="지원 대상", quote="지원 대상: 서울 소재 소상공인")
    assert validate_claims(json.dumps([good]), [hit()])[1]
    for changes in ({"quote": "지원 금액은 9억원입니다"}, {"source_id": "made_up"}, {"field": "선정 확정"}):
        assert validate_claims(json.dumps([{**good, **changes}]), [hit()]) == ([], False)


def test_empty_retries_and_returns_explicit_unknown():
    search = Search([[], []])
    result = engine(search).answer("서울 소상공인 지원 대상은?")
    assert len(search.calls) == 2 and search.calls[0] != search.calls[1]
    assert result["sources"] == [] and "확인할 수 없습니다" in result["answer"]


def test_broad_policy_question_uses_store_details_not_conversation_words():
    queries, region, _ = search_plan("우리 가게가 받을 수 있는 지원금이 있을까요?", "안산", "카페", 2026)
    assert region == "안산"
    assert all("안산" in q and "소상공인" in q for q in queries)
    assert "카페" in queries[0] and "있을까요" not in queries[0]
    assert queries[0] != queries[1]


def test_explicit_ansan_question_overrides_old_seoul_store_setting():
    queries, region, _ = search_plan("안산 소상공인 커피집 사장이 받을 수 있는 지원금이 있나?", "서울", "음식점", 2026)
    assert region == "안산" and "카페" in queries[0]
    assert all("서울" not in q for q in queries)
    assert region_matches("[경기] 소상공인 지원 공고", region)
    assert region_matches("[안산시] 소상공인 지원 공고", region)
    assert not region_matches("[서울] 소상공인 지원 공고", region)


def test_previous_conversation_is_not_used_as_the_current_search_topic():
    question = "다음은 같은 사용자의 이전 대화입니다.\n사용자: 2025년 서울 노란우산\n현재 사용자 질문: 2026년 안산 카페 지원사업 찾아줘"
    queries, region, topics = search_plan(question, "서울", "카페", 2026)
    assert region == "안산" and not topics
    assert all("2025" not in q and "노란우산" not in q for q in queries)


def test_same_official_announcement_is_not_three_different_policies():
    first = hit("https://www.bizinfo.go.kr/sii/siia/selectSIIA200Detail.do?pblancId=PBLN_123")
    second = hit("https://bizinfo.go.kr/sii/siia/selectSIIA200Detail.do?cpage=5&pblancId=PBLN_123&rows=15")
    assert source_identity(first) == source_identity(second)


def test_unofficial_rejected_and_invented_draft_never_served(monkeypatch):
    search = Search([[hit("https://blog.example.com/policy")], [hit()]])
    monkeypatch.setattr("policy_r1.mvp.collect_rollout", lambda *a, **kw: SimpleNamespace(
        finish_reason="answer", answer='[{"source_id":"web_test","field":"지원 대상","quote":"무조건 9억원 지급 확정"}]'))
    result = engine(search).answer("서울 지원 대상과 신청 절차 알려줘")
    assert len(search.calls) == 2
    assert result["sources"][0]["url"] == hit()["url"]
    assert "9억원" not in result["answer"]
    assert result["diagnostics"]["used_extractive_fallback"]
    assert "매장별 최종 자격" in result["answer"]


def test_deadline_does_not_make_search_request():
    search = Search([])
    with pytest.raises(RolloutCancelled):
        engine(search).answer("지원 대상?", deadline=time.monotonic()-1)
    assert not search.calls


def test_government_domain_boundary():
    from policy_r1.official_pages import official_url
    assert official_url("https://www.seoul.go.kr/notice")
    assert not official_url("https://seoul.go.kr.evil.example/notice")
    assert not official_url("https://www.seoul.go.kr@evil.example/notice")


def test_retry_preserves_requested_year_and_excludes_other_region():
    other = {**hit(), "title": "[전남] 2025년 노란우산 지원 공고"}
    search = Search([[other], []])
    result = engine(search).answer("2025년 노란우산 신청 방법은?", region="서울")
    assert all("2025" in q and "2026" not in q for q in search.calls)
    assert not result["sources"]


def test_provider_failure_is_not_reported_as_no_policy():
    from policy_r1.web_search import SearchUnavailable

    class FailedSearch:
        def search(self, *args, **kwargs):
            raise SearchUnavailable("provider failure", code="HTTP_503")

    with pytest.raises(SearchUnavailable):
        engine(FailedSearch()).answer("서울 지원 대상?")


def test_comparison_stops_live_requests_after_quota_error(tmp_path):
    from scripts.evaluate_policy_mvp import FrozenSearch
    from policy_r1.web_search import SearchUnavailable

    class QuotaSearch:
        calls = 0

        def search(self, *args, **kwargs):
            self.calls += 1
            raise SearchUnavailable("quota", code="HTTP_429")

    live = QuotaSearch()
    frozen = FrozenSearch(live, tmp_path)
    for query in ("first", "first", "second"):
        with pytest.raises(SearchUnavailable):
            frozen.search(query, 3)
    assert live.calls == 1


def test_env_loader_does_not_execute_shell(tmp_path, monkeypatch):
    from policy_r1.runtime_env import load_runtime_env
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    marker = tmp_path/"must-not-exist"
    file = tmp_path/"credentials.env"
    file.write_text("SERPAPI_API_KEY='$(touch " + str(marker) + ")'\nUNRELATED_SECRET=ignored\n")
    load_runtime_env(file)
    assert not marker.exists()


def test_grpc_auth_and_source_contract():
    import grpc
    from america.ai.v1 import common_pb2, policy_pb2, policy_pb2_grpc
    from policy_r1.grpc_service import build_server

    class Engine:
        def answer(self, question, **kwargs):
            assert kwargs["region"] == "서울" and kwargs["category"] == "음식점"
            return dict(answer="테스트 응답", sources=[{k:hit()[k] for k in ("title", "url", "snippet", "score")}])

    server, port = build_server(Engine(), "test-only-secret", "127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = policy_pb2_grpc.PolicyAssistantStub(channel)
            request = policy_pb2.PolicyQuestion(question="질문", store=common_pb2.StoreContext(region="서울", category="음식점"))
            with pytest.raises(grpc.RpcError) as error:
                stub.Ask(request, timeout=5)
            assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
            metadata = (("x-internal-key", "test-only-secret"),)
            answer = stub.Ask(request, metadata=metadata, timeout=5)
            assert answer.answer == "테스트 응답" and answer.sources[0].url == hit()["url"]
            chunks = list(stub.AskStream(request, metadata=metadata, timeout=5))
            assert chunks[-1].done and chunks[-1].sources[0].url == answer.sources[0].url
            assert "".join(c.delta for c in chunks) == answer.answer
    finally:
        server.stop(0).wait()
