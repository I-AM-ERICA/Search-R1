import time

import pytest

from policy_r1.inference import ObservedSearch
from policy_r1.rollout import RolloutCancelled
from policy_r1.serving_queries import clean_search_query, broader_search_query, requests_official_sources
from policy_r1.web_search import SearchUnavailable


PREFIX = "기준일: 2026-11-14\n지역: 서울\n업종: 음식점"
ECHO = "기준일: 2026-11-14 지역: 서울 업종: 음식점 소상공인시장진흥공단 정책자금의 공통 지원 자격을 공식 자료로 확인해줘."


@pytest.mark.parametrize("separator", [" ", " / ", "/", " | ", "; ", ", ", "\n"])
def test_echoed_context_delimiters_are_removed(separator):
    question = ECHO.split("음식점 ", 1)[1]
    echo = separator.join([*PREFIX.splitlines(), question])
    assert clean_search_query(echo, PREFIX) == "소상공인시장진흥공단 정책자금 공통 지원 자격"


@pytest.mark.parametrize("before,after", [
    ("2026-11-14", "2025-11-14"), ("서울", "부산"),
    ("음식점", "음식점업"),
])
def test_different_context_values_are_not_removed(before, after):
    echo = " / ".join(PREFIX.replace(before, after).splitlines()) + " / 정책자금"
    assert clean_search_query(echo, PREFIX) == echo


def test_query_constraints_and_urls_survive_context_cleanup():
    query = '2026/11/14 부산 음식점 제외 업종 site:mss.go.kr https://example.org/a/b'
    assert clean_search_query(" / ".join(PREFIX.splitlines()) + " / " + query, PREFIX) == query
    embedded = "정책자금 " + " / ".join(PREFIX.splitlines())
    assert clean_search_query(embedded, PREFIX) == embedded


def test_observed_bad_query_is_cleaned_without_changing_original_constraints():
    cleaned = clean_search_query(ECHO, PREFIX)
    assert cleaned == "소상공인시장진흥공단 정책자금 공통 지원 자격"
    assert broader_search_query(cleaned) == "소상공인시장진흥공단 정책자금 지원"
    specific = '2026년 부산 음식점 제외 업종 site:mss.go.kr "청년 지원"'
    assert clean_search_query(specific, PREFIX) == specific
    assert broader_search_query(specific) is None
    assert clean_search_query("기준일: 2025-01-01 지원 정책", PREFIX).startswith("기준일: 2025")


class Retriever:
    def __init__(self):
        self.calls = []
    def search(self, query, top_k, **kwargs):
        self.calls.append(query)
        return [] if "공통 지원 자격" in query else [{"id": "official", "contents": "공식 안내"}]


def test_empty_result_is_broadened_once_and_total_requests_are_bounded():
    retriever = Retriever()
    observed = ObservedSearch(retriever, lambda: False, time.monotonic()+30, PREFIX, 2)
    hits = observed.search(ECHO, 3)
    assert hits[0]["id"] == "official"
    assert retriever.calls == ["소상공인시장진흥공단 정책자금 공통 지원 자격", "소상공인시장진흥공단 정책자금 지원"]
    assert observed.search(ECHO, 3) == hits  # no repeated API calls
    assert observed.search("새로운 검색", 3) == [] and len(retriever.calls) == 2


def test_no_retry_when_results_exist_or_api_fails_or_request_cancelled():
    retriever = Retriever()
    observed = ObservedSearch(retriever, lambda: False, time.monotonic()+30, PREFIX)
    assert observed.search("지원 정책", 3) and len(retriever.calls) == 1
    class Failing:
        def search(self, *a, **kw):
            raise SearchUnavailable("timeout", code="TIMEOUT")
    observed = ObservedSearch(Failing(), lambda: False, time.monotonic()+30, PREFIX)
    with pytest.raises(SearchUnavailable):
        observed.search(ECHO, 3)
    assert observed.requests == 1
    observed = ObservedSearch(retriever, lambda: True, time.monotonic()+30, PREFIX)
    with pytest.raises(RolloutCancelled):
        observed.search(ECHO, 3)
    assert observed.requests == 0


def test_official_request_rejects_misleading_titles_and_broadens_to_verified_host():
    calls = []
    class Search:
        def search(self, query, top_k, **kwargs):
            calls.append((query, top_k))
            if "공통 지원 자격" in query:
                return [{"id": "private", "title": "소상공인시장진흥공단 공식", "url": "https://consulting.example/semas"},
                        {"id": "spoofed", "title": "공식", "url": "https://www.semas.or.kr.evil.example/"}]
            return [{"id": "real", "url": "https://www.semas.or.kr/notice"}]
    search = ObservedSearch(Search(), lambda: False, time.monotonic()+30, PREFIX, 2, official_only=True)
    assert search.search(ECHO, 3) == [{"id": "real", "url": "https://www.semas.or.kr/notice"}]
    assert len(calls) == 2 and all(top_k == 5 for _, top_k in calls)
    assert search.selection_log[0]["rejected_unverified_domain"] == 2
    assert set(search.hits) == {"real"}
    assert requests_official_sources("공식 자료로 확인해줘")
    assert not requests_official_sources("비공식 자료")
    assert not requests_official_sources("다음은 같은 사용자의 이전 대화입니다. 공식 자료\n현재 사용자 질문: 일반 검색")
