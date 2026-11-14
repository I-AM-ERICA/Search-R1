import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from policy_r1 import trainer
from policy_r1.recovery import save_interrupted_collection
from policy_r1.rollout import Trajectory
from policy_r1.sqlite_search import SQLiteRetriever, build_sqlite_index
from scripts.pause_for_checkpoint import safe_collection
from scripts import pause_for_checkpoint
from test_training_signal import make_experiment


@pytest.mark.parametrize("rows", [3, 4])
def test_interrupt_collection_replays_rows_and_saves_completed_weights(tmp_path, monkeypatch, rows):
    experiment = make_experiment(tmp_path, monkeypatch, format_reward=0.1)
    experiment.config.train_sampling = "shuffle"
    experiment.train_data = experiment.train_data[:rows]
    experiment.config.save_every = 10
    experiment.config.eval_every = 100
    database = tmp_path / "index.sqlite"
    build_sqlite_index("data/demo/corpus.jsonl", database)
    experiment.retriever = SQLiteRetriever(database)
    calls = 0
    def search(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(experiment.retriever, "_search", search)
    def rollout(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:  # One complete update, then two consumed rows.
            experiment.retriever.search("test")
        return Trajectory([1, 5, 6, 7, 8, 9], [False, False, True, True, True, True],
                          "answer", 0, "answer", [])
    monkeypatch.setattr(trainer, "collect_rollout", rollout)
    try:
        experiment.train()
    except KeyboardInterrupt as exc:
        assert experiment.step == 1
        assert experiment.train_examples_seen == 4
        weights = {n: p.detach().clone() for n, p in experiment.actor.named_parameters()}
        saved = save_interrupted_collection(experiment, exc, exc.__traceback__)
    else:
        pytest.fail("No interrupt")
    assert experiment.train_examples_seen == 2
    assert len(experiment.train_order) == (4 if rows == 3 else 2)
    assert all(torch.equal(weights[n], p) for n, p in experiment.actor.named_parameters())
    state = torch.load(saved / "training_state.pt", weights_only=True)
    assert state["step"] == 1 and state["train_examples_seen"] == 2
    assert state["train_order"] == experiment.train_order
    assert len(state["actor_optimizer"]["state"]) > 0
    assert json.loads((saved / "recovery.json").read_text())["replayed_rows"] == 2
    assert experiment.train()[0]["step"] == 2
    experiment.retriever.close()


def test_recovery_refuses_unknown_interrupt_without_mutation(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch)
    experiment.config.train_sampling = "shuffle"
    with pytest.raises(RuntimeError, match="KeyboardInterrupt"):
        save_interrupted_collection(experiment, ValueError(), None)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt as exc:
        with pytest.raises(RuntimeError, match="rollout collection"):
            save_interrupted_collection(experiment, exc, exc.__traceback__)
    assert experiment.train_examples_seen == 0
    assert not list(experiment.output.glob("checkpoint-*"))


def test_recovery_cell_reads_ipython_last_traceback(tmp_path, monkeypatch):
    from IPython.core.interactiveshell import InteractiveShell
    experiment = make_experiment(tmp_path, monkeypatch, format_reward=0.1)
    experiment.config.train_sampling = "shuffle"
    experiment.config.save_every = 10
    database = tmp_path / "index.sqlite"
    build_sqlite_index("data/demo/corpus.jsonl", database)
    experiment.retriever = SQLiteRetriever(database)
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(experiment.retriever, "_search", interrupt)
    calls = 0
    def rollout(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            experiment.retriever.search("test")
        return Trajectory([1, 5, 6, 7, 8, 9], [False, False, True, True, True, True],
                          "answer", 0, "answer", [])
    monkeypatch.setattr(trainer, "collect_rollout", rollout)
    shell = InteractiveShell(user_ns={"experiment": experiment}, ipython_dir=str(tmp_path / "ipython"))
    result = shell.run_cell("history = experiment.train()")
    assert isinstance(result.error_in_exec, KeyboardInterrupt)
    recovered = shell.run_cell("%run -i scripts/recover_notebook.py")
    assert recovered.success
    assert experiment.train_examples_seen == 2
    assert (experiment.output / "checkpoint-00001/training_state.pt").is_file()
    assert (experiment.output / "RECOVERED_CHECKPOINT").read_text().strip().endswith("checkpoint-00001")
    experiment.retriever.close()


def test_stop_marker_saves_completed_update_and_skips_evaluation(tmp_path, monkeypatch):
    experiment = make_experiment(tmp_path, monkeypatch, format_reward=0.1)
    experiment.config.eval_every = 1
    def rollout(*args, **kwargs):
        (experiment.output / "STOP_AFTER_STEP").touch()
        return Trajectory([1, 5, 6, 7, 8, 9], [False, False, True, True, True, True],
                          "answer", 0, "answer", [])
    monkeypatch.setattr(trainer, "collect_rollout", rollout)
    monkeypatch.setattr(experiment, "evaluate", lambda: pytest.fail("Should skip evaluation"))
    history = experiment.train()
    assert len(history) == 1 and experiment.step == 1
    assert (experiment.output / "checkpoint-00001/training_state.pt").is_file()
    assert history[0]["collect_seconds"] >= 0
    assert history[0]["update_seconds"] >= 0
    assert experiment.train() == []


def test_safe_snapshot_requires_main_thread_collection_and_search():
    stack = ('Thread 42 (active): "MainThread"\n'
             '    SQLiteRetriever.search (policy_r1/sqlite_search.py:161)\n'
             '    PPOExperiment._collect_batch (policy_r1/trainer.py:178)\n'
             '    PPOExperiment.train (policy_r1/trainer.py:318)\n'
             'Thread 43 (idle): "IOPub"\n')
    assert safe_collection(stack, 42)
    assert not safe_collection(stack, 43)
    assert not safe_collection(stack.replace("_collect_batch", "_update"), 42)
    assert not safe_collection(stack.replace('"MainThread"', '"worker"'), 42)


@pytest.mark.parametrize("snapshot_fails", [False, True])
def test_pause_helper_rechecks_frozen_stack_and_always_resumes(monkeypatch, snapshot_fails):
    stack = ('Thread 42 (active): "MainThread"\n'
             '    SQLiteRetriever.search (policy_r1/sqlite_search.py:161)\n'
             '    PPOExperiment._collect_batch (policy_r1/trainer.py:178)\n'
             '    PPOExperiment.train (policy_r1/trainer.py:318)\n')
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["pause_for_checkpoint.py", "--pid", "42"])
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"python\0-m\0ipykernel_launcher")
    monkeypatch.setattr(Path, "read_text", lambda path: "Name:\tpython\nState:\tT (stopped)\n")
    monkeypatch.setattr(pause_for_checkpoint, "start_time", lambda pid: "123")
    signals = []
    monkeypatch.setattr(pause_for_checkpoint.os, "kill", lambda pid, sig: signals.append(sig))
    snapshots = 0
    def run(command, **kwargs):
        nonlocal snapshots
        snapshots += 1
        assert "--nonblocking" in command
        if snapshots == 2:
            assert signals == [signal.SIGSTOP]
            if snapshot_fails:
                raise subprocess.TimeoutExpired(command, 5)
        return SimpleNamespace(stdout=stack)
    monkeypatch.setattr(subprocess, "run", run)
    ended = []
    watchdog = SimpleNamespace(terminate=lambda: ended.append("terminate"), wait=lambda: ended.append("wait"))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: watchdog)
    if snapshot_fails:
        with pytest.raises(subprocess.TimeoutExpired):
            pause_for_checkpoint.main()
        assert signals == [signal.SIGSTOP, signal.SIGCONT]
    else:
        pause_for_checkpoint.main()
        assert signals == [signal.SIGSTOP, signal.SIGINT, signal.SIGCONT]
    assert snapshots == 2
    assert ended == ["terminate", "wait"]
