from policy_r1.official_pages import extract_html
from policy_r1.evidence_focus import focus_official_excerpt


def sample():
    html = ('<div class="contents"><h4>가상 정책 안내</h4>'
            '<dl><dt>융자대상</dt><dd><strong>공통 지원 자격</strong>'
            '<ul><li>이 예시는 자동 테스트용 가상 자료이며 지원 자격은 해당 공고에서 확인합니다.</li>'
            '<li>제외업종과 예외 조건도 함께 확인합니다.</li></ul>'
            '<strong>세부 지원요건</strong><table><tr><td>다른 사업의 상세 표 내용입니다.</td></tr></table>'
            '<strong>금리 안내</strong><p>다른 항목의 금리 표와 신청 절차 안내입니다.</p></dd></dl></div>')
    return extract_html(html.encode())


def hit(text):
    return {"id":"official", "title":"가상 자료", "url":"https://www.semas.or.kr/notice",
            "snippet":text, "contents":text, "evidence_kind":"official_page_excerpt"}


def test_heading_boundaries_and_verbatim_focus_keep_exclusions():
    text = sample()
    assert '\n\n공통 지원 자격\n' in text and '\n\n세부 지원요건\n' in text
    original = hit(text)
    result = focus_official_excerpt(original, '기관 정책자금 공통 지원 자격')
    assert result['snippet'] in text
    assert '제외업종과 예외 조건' in result['snippet']
    assert '금리 안내' not in result['snippet'] and '세부 지원요건' not in result['snippet']
    assert original['snippet'] == text and result is not original


def test_selects_by_query_and_does_not_modify_snippets_or_unstructured_text():
    text = sample()
    original = hit(text)
    assert focus_official_excerpt(original, '존재하지않는 주제') is original
    snippet = {**original, 'evidence_kind':'search_snippet'}
    assert focus_official_excerpt(snippet, '공통 지원 자격') is snippet
    single = hit('공통 지원 자격 ' * 30)
    assert focus_official_excerpt(single, '공통 지원 자격') is single


def test_neighboring_exclusion_section_is_preserved():
    relevant = '공통 지원 자격\n' + '가상 지원 자격 및 공고 확인 안내입니다. ' * 4
    exclusions = '제외 조건\n가상 제외 조건도 반드시 확인합니다.'
    text = '개요\n가상 문서 안내\n\n' + relevant + '\n\n' + exclusions + '\n\n다른 항목\n관련 없는 표'
    result = focus_official_excerpt(hit(text), '공통 지원 자격')
    assert result['snippet'] == relevant.strip() + '\n\n' + exclusions
