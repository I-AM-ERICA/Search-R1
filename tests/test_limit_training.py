import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.limit_training import latest_checkpoint, resume_command, select_run
from scripts import limit_training


def test_select_only_active_run_and_reject_ambiguity(tmp_path):
    def make(name, mtime):
        run = tmp_path / "outputs" / name
        run.mkdir(parents=True)
        config = run / "config.json"
        config.write_text(json.dumps({"output_dir": f"outputs/{name}"}))
        os.utime(config, (mtime, mtime))
        (run / "metrics.jsonl").write_text('{"step":140}\n')
        return run
    make("yesterday", 100)
    current = make("current", 1000)
    assert select_run(tmp_path, 999, []) == current
    make("ambiguous", 1001)
    with pytest.raises(RuntimeError, match="uniquely"):
        select_run(tmp_path, 999, [])
    assert select_run(tmp_path, 999, [], "outputs/current") == current


def test_latest_complete_checkpoint_and_exact_cap(tmp_path):
    (tmp_path / "checkpoint-00150").mkdir()
    saved = tmp_path / "checkpoint-00140"
    for name in ["config.json", "metadata.json", "training_state.pt",
                 "actor/adapter_model.safetensors", "critic/adapter_model.safetensors"]:
        path = saved / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    assert latest_checkpoint(tmp_path) == (140, saved)
    command = resume_command("python", tmp_path, saved, tmp_path / "new", 150)
    assert command[command.index("--total-updates") + 1] == "150"
    assert "--skip-final-eval" in command and "--skip-resume-checks" in command


def test_controller_saves_then_resumes_to_target(tmp_path, monkeypatch):
    def checkpoint(run, step):
        path = run / f"checkpoint-{step:05d}"
        for name in ["config.json", "metadata.json", "training_state.pt",
                     "actor/adapter_model.safetensors", "critic/adapter_model.safetensors"]:
            file = path / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("test")
        return path
    run = tmp_path / "outputs/current"
    saved = checkpoint(run, 140)
    (run / "metrics.jsonl").write_text('{"step":140}\n')
    monkeypatch.setattr(limit_training, "__file__", str(tmp_path / "scripts/limit_training.py"))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["limit_training.py", "--target", "150"])
    monkeypatch.setattr(limit_training, "active_training", lambda root: (42, 1, 0, []))
    monkeypatch.setattr(limit_training, "select_run", lambda *args: run)
    resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path, *args, **kwargs:
                        tmp_path if str(path) == "/proc/42/cwd" else resolve(path, *args, **kwargs))
    states = iter([(1, 0), None])
    def process(pid):
        state = next(states)
        if state is None:
            assert (run / "STOP_AFTER_STEP").exists()
            raise FileNotFoundError()
        return state
    monkeypatch.setattr(limit_training, "process_start", process)
    monkeypatch.setattr(limit_training.subprocess, "check_output", lambda *args, **kwargs: "--skip-final-eval")
    def launch(command, **kwargs):
        assert command[command.index("--resume-from") + 1] == str(saved)
        assert command[command.index("--total-updates") + 1] == "150"
        assert "--skip-final-eval" in command
        checkpoint(Path(command[command.index("--output-dir") + 1]), 150)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(limit_training.subprocess, "run", launch)
    limit_training.main()
    assert len(list((tmp_path / "outputs").glob("search_r1_until_00150_*/checkpoint-00150"))) == 1
