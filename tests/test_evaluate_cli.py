import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import evaluate


def test_subset_comparison_selects_same_questions_and_keeps_checkpoint_config(tmp_path, monkeypatch):
    checkpoints = [tmp_path / "before", tmp_path / "after"]
    observed = []
    class Experiment:
        def __init__(self, config, resume_from):
            self.output = Path(config.output_dir)
            self.output.mkdir()
            self.val_data = [{"id": str(i)} for i in range(100)]
            self.eval_data = self.val_data[:50]
        def evaluate(self, *, full, split):
            assert full is False and split == "val"
            observed.append([row["id"] for row in self.eval_data])
            return {"eval_count": len(self.eval_data)}
    monkeypatch.setitem(sys.modules, "policy_r1.trainer", SimpleNamespace(PPOExperiment=Experiment))
    for i, checkpoint in enumerate(checkpoints):
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"eval_max_samples": 200}')
        output = tmp_path / f"eval-{i}"
        monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", str(checkpoint),
                                         "--max-samples", "30", "--output-dir", str(output)])
        evaluate.main()
        assert (checkpoint / "config.json").read_text() == '{"eval_max_samples": 200}'
        manifest = json.loads((output / "evaluation_sample.json").read_text())
        assert manifest["ids"] == observed[-1]
    assert observed[0] == observed[1]
    assert len(set(observed[0])) == 30


@pytest.mark.parametrize("flags", [["--max-samples", "-1"], ["--split", "test", "--max-samples", "30"]])
def test_reject_invalid_subset_before_loading_model(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", "/missing", *flags])
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2
