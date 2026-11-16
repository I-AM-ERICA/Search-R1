import html
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from policy_r1 import official_pages as pages
from policy_r1.web_search import SerpAPIRetriever


def test_editor_html_decoded_as_inert_article_and_navigation_not_selected():
    body = ('<h1>공고 제목</h1><p>모집기간: 2026.09.28 ~ 2026.10.16</p>'
            '<p>본점 주소지가 서울인 소상공인. 세부 사항은 첨부 공고 참조.</p>' * 4
            + '<script>secret()</script><nav>DO NOT INCLUDE</nav>'
            + '<textarea id="editContents">NOT AN ARTICLE</textarea>')
    raw = ('<main><div>사이트 메뉴 반복 ' + '메뉴 ' * 200 + '</div><div class="board_view">'
           '<div>등록일 2026.09.28</div><textarea id="editContents" style="display:none">'
           + html.escape(body) + '</textarea></div></main>').encode()
    text = pages.extract_html(raw)
    assert '모집기간: 2026.09.28 ~ 2026.10.16' in text
    for bad in ('<h1>', '&lt;', '사이트 메뉴', 'secret()', 'DO NOT INCLUDE', 'NOT AN ARTICLE'):
        assert bad not in text
    assert '등록일 2026.09.28' in text


def test_gcgf_content_container_is_recognized_without_taking_navigation():
    body = '공식 본문: 접수완료 안내 및 진행단계별 안내를 제공합니다. ' * 6
    raw = ('<div id="subContent"><div>사이트 메뉴</div><div class="subCntBody">'
           '<div id="contsView"><h3>고객 안내</h3><p>' + body + '</p></div></div></div>').encode()
    text = pages.extract_html(raw)
    assert body.strip() in text and '사이트 메뉴' not in text
    with pytest.raises(pages.PageUnavailable, match='no_html_body'):
        pages.extract_html(b'<main><nav>menu only</nav><textarea>hidden notes</textarea></main>')


def test_safe_http_and_transport_details_survive_to_per_url_candidate_trace(monkeypatch):
    monkeypatch.setattr(pages.shutil, 'which', lambda _: '/usr/bin/curl')
    monkeypatch.setattr(pages.socket, 'getaddrinfo', lambda *a, **kw: [(None,None,None,None,('1.1.1.1',443))])
    payload = {'web_results': [dict(title='공식 문서', link='https://www.bizinfo.go.kr/notice', snippet=None)]}
    for code, output, details in [(0,b'302\ntext/html', {'http_status':302,'content_type':'text/html'}),
                                 (0,b'200\napplication/pdf', {'http_status':200,'content_type':'application/pdf'}),
                                 (60,b'000\n', {'curl_exit_code':60})]:
        monkeypatch.setattr(pages.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=code,stdout=output,stderr=b'secret-cookie'))
        search = SerpAPIRetriever('test-key', engine='naver', cache_size=0, require_page_evidence=True,
            official_only=True, opener=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
        assert search.search('주제',3) == []
        trace = search.last_diagnostics['candidate_trace'][0]
        assert trace['status'] == 'page_unavailable'
        assert trace['failure_details'] == details
        assert 'secret-cookie' not in json.dumps(search.last_diagnostics)
        assert trace['url'] == payload['web_results'][0]['link']


def test_rejected_domains_and_unattempted_rows_do_not_become_fake_http_failures():
    rows = [dict(title='문서',link=f'https://www.bizinfo.go.kr/{i}',snippet=None) for i in range(5)]
    rows.append(dict(title='민간',link='https://example.org/notice',snippet='snippet'))
    search = SerpAPIRetriever('dummy', engine='naver', cache_size=0, require_page_evidence=True,
        official_only=True, page_fetcher=lambda *a, **kw: '가상 정책 본문입니다. ' * 20,
        opener=lambda *a, **kw: io.BytesIO(json.dumps({'web_results':rows}).encode()))
    assert len(search.search('조건', 3)) == 3
    trace = search.last_diagnostics['candidate_trace']
    assert [r['status'] for r in trace] == ['accepted']*3 + ['page_budget']*2 + ['unverified_domain']
    assert all('failure_details' not in r for r in trace)


def test_attachments_do_not_exhaust_three_html_reads_and_titles_are_not_url_types():
    rows = [dict(title='지원사업', link='https://www.bizinfo.go.kr/cmm/fms/fileDown.do?atchFileId=A'),
            dict(title='지원사업', link='https://www.bizinfo.go.kr/cmm/fms/getImageFile.do?atchFileId=B'),
            dict(title='지원사업', link='https://www.mss.go.kr/common/board/Download.do?bcIdx=1')]
    rows += [dict(title='공고.hwp 첨부 안내', link=f'https://www.bizinfo.go.kr/notice/{i}') for i in range(4)]
    calls = []
    search = SerpAPIRetriever('dummy', engine='naver', cache_size=0, require_page_evidence=True,
        official_only=True, page_fetcher=lambda url, **kw: calls.append(url) or '지원 대상과 신청 조건의 가상 본문입니다. ' * 10,
        opener=lambda *a, **kw: io.BytesIO(json.dumps({'web_results':rows}).encode()))
    assert len(search.search('조건',3)) == 3
    assert calls == [r['link'] for r in rows[3:6]]
    assert search.last_diagnostics['page_fetch_attempts'] == 3
    assert search.last_diagnostics['rejected'] == {'unsupported_resource':3,'page_budget':1}
    assert pages.unsupported_document_url('https://example.org/file%2Epdf?q=1')
    assert not pages.unsupported_document_url('https://example.org/notice?q=manual.pdf')
    with pytest.raises(pages.PageUnavailable, match='unsupported_resource'):
        pages.fetch_public_page(rows[0]['link'])  # no network needed to classify


def test_recorded_provider_replay_only_reads_unseen_html_and_preserves_source(tmp_path):
    from scripts.collect_policy_curriculum import collect
    from scripts.replay_policy_pages import replay
    from scripts.prepare_policy_sft import load_capture
    import hashlib
    trace = [dict(rank=i+1, url=f'https://www.bizinfo.go.kr/cmm/fms/fileDown.do?atchFileId={i}',
                  title='첨부', official_host=True,status='page_unavailable',failure_reason='not_html_200',
                  failure_details={'http_status':403}) for i in range(3)]
    url='https://www.bizinfo.go.kr/sii/siia/selectSIIA200Detail.do?pblancId=TEST'
    trace.append(dict(rank=4,url=url,title='실제 본문',official_host=True,status='page_budget'))
    class Search:
        engine,timeout='naver',30
        last_diagnostics={'result_count':4,'page_fetch_attempts':3,'candidate_trace':trace}
        def search(self,*a,**kw):return []
    source=tmp_path/'source'
    collect({'cases':[{'id':'case','candidate_queries':['질문']}]},Search(),source)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    with pytest.raises(RuntimeError,match='Unrecorded HTML'):
        replay(source,tmp_path/'missing')
    calls=[]
    result=replay(source,tmp_path/'replayed',fetch_missing=True,max_live_pages=1,
                  fetcher=lambda u,**kw:calls.append(u) or '가상 근거 본문입니다. '*20)
    assert calls==[url] and result[0]['after_bodies']==1 and result[0]['before_bodies']==0
    _,snapshots,manifest=load_capture(tmp_path/'replayed')
    assert manifest['unique_provider_requests']==0 and manifest['live_missing_page_requests']==1
    assert next(iter(snapshots.values()))['provider_replayed']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
