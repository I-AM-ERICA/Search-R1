import io
import json
from dataclasses import replace

import pytest

from policy_r1.official_pages import PageUnavailable, public_page_url, fetch_public_page
from policy_r1.page_summary import previous_summary_urls
from policy_r1.inference import InferenceFailure
from policy_r1.web_search import SerpAPIRetriever
from test_web_serving import engine


URL = "https://www.semas.or.kr/notice"
BODY = "가상 사업의 지원 내용: 음식점 대상 상담을 지원합니다. 신청은 공고의 접수처에서 합니다. " * 5
PAYLOAD = {"organic_results": [{"title": "가상 사업", "link": URL, "snippet": "검색 요약만으로는 세부 내용을 알 수 없습니다."}]}
HISTORY = "다음은 같은 사용자의 이전 대화입니다.\n사용자: 음식점 지원 정책\n정책 비서: 안내입니다.\n이전 답변 출처 URL: " + URL + "\n\n현재 사용자 질문: 어떤 내용이야? 링크 말고 요약해줘"


def retriever(fetcher, payload=PAYLOAD):
    return SerpAPIRetriever("dummy", require_page_evidence=True, page_fetcher=fetcher,
                           opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))


def test_snippets_are_replaced_with_read_bodies_and_reused_for_followups():
    calls = []
    def fetch(url, **kw):
        calls.append(url)
        return BODY
    search = retriever(fetch)
    hits = search.search("가상 지원")
    assert calls == [URL] and hits[0]["snippet"] == BODY.strip()
    assert hits[0]["evidence_kind"] == "official_page_excerpt"
    assert "검색 요약만으로는" not in hits[0]["contents"]
    assert search.read_pages([URL])[0]["snippet"] == BODY.strip()
    assert calls == [URL]


def test_unreadable_page_is_not_replaced_with_snippet_or_previous_answer():
    def fail(*a, **kw):
        raise PageUnavailable("not_html_200")
    search = retriever(fail)
    assert search.search("가상 지원") == []
    assert search.last_diagnostics["rejected"] == {"page_unavailable": 1}
    service = engine([])
    service.retriever = search
    with pytest.raises(InferenceFailure, match="Unable to read previous"):
        service.answer(HISTORY)


def test_public_html_reads_still_block_private_addresses_and_redirect_wrappers():
    for url in ["https://127.0.0.1/", "http://www.semas.or.kr/", "https://example.local/", "https://u:p@example.com/", "https://example.com:8080/"]:
        assert not public_page_url(url)
    assert public_page_url("https://public.example.org/policy")
    calls = []
    search = retriever(lambda *a, **kw: calls.append(a) or BODY)
    assert search.read_pages(["https://www.google.com/goto?url=encoded", "https://127.0.0.1/"]) == []
    assert not calls


def test_generic_public_host_cannot_resolve_to_private_ip(monkeypatch):
    from policy_r1 import official_pages as pages
    monkeypatch.setattr(pages.shutil, "which", lambda _: "/usr/bin/curl")
    monkeypatch.setattr(pages.socket, "getaddrinfo", lambda *a, **kw: [(None, None, None, None, ("10.0.0.1", 443))])
    with pytest.raises(PageUnavailable, match="nonpublic_address"):
        fetch_public_page("https://public.example.org/")


def test_body_fetches_are_bounded_and_official_pages_are_prioritized():
    rows = [{"title": str(i), "link": f"https://public.example.org/{i}", "snippet": "summary"} for i in range(8)]
    rows.append(PAYLOAD["organic_results"][0])
    calls = []
    search = retriever(lambda url, **kw: calls.append(url) or BODY, {"organic_results": rows})
    hits = search.search("지원", 5)
    assert len(hits) == len(calls) == 3 and calls[0] == URL


def test_followup_uses_page_observation_without_searching_vague_question():
    service = engine(["<answer>가상 사업은 음식점 대상 상담을 지원합니다. 공고의 접수처로 신청합니다.</answer>"])
    service.retriever = retriever(lambda *a, **kw: BODY)
    service.retriever._open = lambda *a, **kw: pytest.fail("should read previous URL, not search vague followup")
    result = service.answer(HISTORY)
    assert result["sources"][0]["url"] == URL
    assert "상담" in result["answer"] and result["search_requests"] == 0
    assert result["evidence_mode"] == "page_excerpts"
    assert previous_summary_urls("일반 질문에 있는 " + URL) == []
    assert previous_summary_urls(HISTORY.replace("어떤 내용이야? 링크 말고 요약해줘", "다른 지역 창업 지원을 찾아줘")) == []


def test_pointer_only_answer_gets_one_grounded_rewrite():
    service = engine(["<search>음식점 지원</search>", "<answer>소상공인 정책 정보입니다.</answer>",
                      "<answer>가상 사업은 음식점 대상 상담을 지원합니다.</answer>"])
    service.retriever = retriever(lambda *a, **kw: BODY)
    result = service.answer("음식점 지원을 찾아줘")
    assert "상담" in result["answer"] and result["search_requests"] == 1
    assert result["searches"] == 1


def test_multiple_sources_reach_model_with_body_text_and_links():
    from scripts.diagnose_policy import probe
    payload = {"organic_results": [*PAYLOAD["organic_results"], {"title": "다른 공고", "link": "https://public.example.org/2", "snippet": "요약"}]}
    service = engine(["<search>지원</search>", "<answer>두 공고 모두 상담 지원 내용을 안내합니다.</answer>"])
    service.retriever = retriever(lambda *a, **kw: BODY * 10, payload)
    result = probe(service, "지원", show_answer=True)
    assert result["status"] == "success" and result["source_count"] == 2
    assert "읽은 HTML 본문" in result["trajectory"]["events"][1]["preview"]
