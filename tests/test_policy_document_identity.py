from policy_r1.policy_sft import document_key


def test_bizinfo_notice_tracking_variants_share_identity():
    base = 'https://www.bizinfo.go.kr/sii/siia/selectSIIA200Detail.do'
    assert document_key(base+'?cpage=6&pblancId=PBLN_0000123&cat=') == document_key(base+'?pblancId=PBLN_0000123')
    assert document_key(base+'?pblancId=PBLN_0000124') != document_key(base+'?pblancId=PBLN_0000123')


def test_other_hosts_paths_and_ambiguous_ids_not_collapsed():
    base = 'https://www.bizinfo.go.kr/sii/siia/selectSIIA200Detail.do'
    assert document_key(base+'?pblancId=PBLN_1&pblancId=PBLN_2') != document_key(base+'?pblancId=PBLN_1')
    other = base.replace('www.bizinfo.go.kr', 'example.com')
    assert document_key(other+'?pblancId=PBLN_1&cpage=6') != document_key(other+'?pblancId=PBLN_1')
