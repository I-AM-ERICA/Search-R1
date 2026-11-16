import pytest
from scripts.probe_sft_checkpoints import verify_source, selected_inputs


def test_reference_checks_reject_changed_files_and_path_escape(tmp_path):
    (tmp_path/'run.json').write_text('{}')
    with pytest.raises(ValueError, match='Source file changed'):
        verify_source(tmp_path, {'files':{'run.json':'incorrect'}})
    with pytest.raises(ValueError, match='escapes'):
        verify_source(tmp_path, {'files':{'../outside':'incorrect'}})


def test_explicit_stop_probe_subset_cannot_add_unseen_inputs_or_duplicates():
    rows = [dict(case_id='a',stage='initial',input_ids=[1]),
            dict(case_id='b',stage='final_answer',input_ids=[1,2])]
    assert selected_inputs(rows, {}) == rows
    assert selected_inputs(rows, {'probe_keys':[['b','final_answer']]}) == [rows[1]]
    for keys in ([], [['c','initial']], [['b','final_answer'],['b','final_answer']]):
        with pytest.raises(ValueError, match='Invalid probe subset'):
            selected_inputs(rows, {'probe_keys':keys})
