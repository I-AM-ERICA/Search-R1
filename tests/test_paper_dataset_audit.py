import pytest
from scripts.audit_paper_dataset import analyze, read_jsonl
from policy_r1.data import question_key


def row(q, source='hotpotqa'):
    return dict(question=q, data_source=source, answers=['answer'])


def test_jsonl_preserves_unicode_line_separator_in_question(tmp_path):
    import json
    path = tmp_path / 'rows.jsonl'
    record = row('question\u2028continued')
    path.write_text(json.dumps(record, ensure_ascii=False)+'\n')
    assert read_jsonl(path) == [record]


def test_overlap_and_validation_selection_exclude_inspected_cases():
    prepared = dict(train=[row('Training?')], val=[row('Seen'), row('Fresh'), row('NQ', 'nq')], test=[row('Test')])
    raw = dict(train={question_key(q): row(q) for q in ['training', 'Seen', 'Fresh', 'New']},
               test={question_key('Test'): row('Test')})
    report, cases = analyze(raw, prepared, [row('seen!')])
    assert report['overlap']['train']['overlap_with_prepared'] == dict(train=1, val=2, test=0)
    assert report['overlap']['train']['not_in_any_prepared_split'] == 1
    assert [r['question'] for r in cases] == ['Fresh']


def test_fail_closed_on_existing_split_leakage():
    with pytest.raises(ValueError, match='overlap'):
        analyze(dict(train={}, test={}), dict(train=[row('same')], val=[row('Same?')], test=[]), [])


def test_selection_does_not_use_answer_labels():
    prepared = dict(train=[], val=[row(str(i)) for i in range(10)], test=[])
    _, a = analyze(dict(train={}, test={}), prepared, [], 4)
    for r in prepared['val']:
        r['answers'] = ['changed label']
    prepared['val'].reverse()
    _, b = analyze(dict(train={}, test={}), prepared, [], 4)
    assert [r['question'] for r in a] == [r['question'] for r in b]
