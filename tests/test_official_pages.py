import io
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from policy_r1 import official_pages as pages
from policy_r1.web_search import SerpAPIRetriever


URL = 'https://www.semas.or.kr/web/SUP01/SUP0103/SUP010301.kmdc'
BODY = '공식 공고의 대상과 신청 기간은 별도 확인해야 합니다. ' * 10


def test_content_extraction_excludes_navigation_and_scripts():
    html = ('<div id="content"><nav>내비게이션</nav><div class="contents"><h4>정책 공고</h4>'
            '<script>malicious text</script><form>로그인</form><p hidden>숨김</p>'
            '<p>' + BODY + '</p><br><p>추가 내용</p></div><footer>꼬리말</footer></div>')
    result = pages.extract_html(html.encode())
    assert '정책 공고' in result and '추가 내용' in result
    for excluded in ('내비게이션', 'malicious', '로그인', '숨김', '꼬리말'):
        assert excluded not in result
    with pytest.raises(pages.PageUnavailable, match='no_html_body'):
        pages.extract_html(b'<html><nav>Menu only</nav></html>')


@pytest.mark.parametrize('url', ['http://www.semas.or.kr/', 'https://www.semas.or.kr.evil.example/',
    'https://127.0.0.1/', 'https://www.semas.or.kr:444/', 'https://user@www.semas.or.kr/'])
def test_only_supported_https_hosts(url):
    assert not pages.official_url(url)
    with pytest.raises(pages.PageUnavailable, match='unsupported_host'):
        pages.fetch_official_page(url)


def test_public_ip_is_pinned_no_credentials_or_redirects(monkeypatch):
    monkeypatch.setattr(pages.shutil, 'which', lambda x: '/usr/bin/curl')
    monkeypatch.setattr(pages.socket, 'getaddrinfo', lambda *a, **kw: [(None,None,None,None,('1.1.1.1',443))])
    def run(args, **kwargs):
        assert args[1] == '--disable' and '--location' not in args and '-k' not in args
        assert args[args.index('--resolve') + 1] == 'www.semas.or.kr:443:1.1.1.1'
        assert args[args.index('--noproxy') + 1] == '*'
        assert 'SERPAPI_API_KEY' not in kwargs['env']
        Path(args[args.index('--output') + 1]).write_bytes(('<main>'+BODY+'</main>').encode())
        return SimpleNamespace(returncode=0,stdout=b'200\ntext/html; charset=utf-8')
    monkeypatch.setattr(pages.subprocess, 'run', run)
    assert BODY.strip() in pages.fetch_official_page(URL)
    monkeypatch.setattr(pages.socket, 'getaddrinfo', lambda *a, **kw: [(None,None,None,None,('127.0.0.1',443))])
    with pytest.raises(pages.PageUnavailable, match='nonpublic_address'):
        pages.fetch_official_page(URL)


def test_redirect_and_pdf_are_not_treated_as_html(monkeypatch):
    monkeypatch.setattr(pages.shutil,'which',lambda x:'/usr/bin/curl')
    monkeypatch.setattr(pages.socket,'getaddrinfo',lambda *a,**kw:[(None,None,None,None,('1.1.1.1',443))])
    for header in (b'302\ntext/html', b'200\napplication/pdf'):
        monkeypatch.setattr(pages.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout=header))
        with pytest.raises(pages.PageUnavailable,match='not_html_200'):
            pages.fetch_official_page(URL)


def test_real_response_shape_official_body_fallback_and_failure():
    payload = {'web_results': [
        {'title':'지원사업 - 소상공인시장진흥공단','link':URL,'snippet':None},
        {'title':'소상공인시장진흥공단','link':'https://namu.wiki/w/example','snippet':'기관 소개'}],
        'inline_images_results':[{'title':'블로그','link':'https://blog.naver.com/example'}]}
    calls=[]
    def fetch(url,timeout):
        calls.append(url)
        return BODY
    search=SerpAPIRetriever('dummy',engine='naver',cache_size=0,page_fetcher=fetch,
        opener=lambda *a,**kw:io.BytesIO(json.dumps(payload).encode()))
    hits=search.search('정책자금')
    assert calls==[URL] and len(hits)==2
    assert hits[0]['evidence_kind']=='official_page_excerpt' and '공식 HTML 본문 발췌' in hits[0]['contents']
    assert hits[1]['evidence_kind']=='search_snippet'
    def fail(*a,**kw):
        raise pages.PageUnavailable('page_request_failed')
    search._fetch_page=fail
    hits=search.search('정책자금')
    assert len(hits)==1 and hits[0]['url'].startswith('https://namu.wiki')
    assert search.last_diagnostics['page_fetch_failures']=={'page_request_failed':1}
