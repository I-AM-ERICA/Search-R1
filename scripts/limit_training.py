"""Linux: safely stop an active CLI run, then resume to an exact total update cap.

Run this controller in its own tmux session. No signals are sent to training:
the existing STOP_AFTER_STEP mechanism finishes and saves the current update.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_start(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    ticks = int(text.rsplit(")", 1)[1].split()[19])
    boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                if line.startswith("btime "))
    return ticks, boot + ticks / os.sysconf("SC_CLK_TCK")


def active_training(root):
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode().strip("\0").split("\0")
            cwd = (entry / "cwd").resolve(strict=True)
            scripts = [arg for arg in argv[1:] if Path(arg).name == "train.py"]
            if not any((cwd / arg).resolve() == root / "scripts/train.py" for arg in scripts):
                continue
            ticks, started = process_start(int(entry.name))
            matches.append((int(entry.name), ticks, started, argv))
        except (OSError, ValueError, StopIteration, UnicodeError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one active scripts/train.py in this project; found {len(matches)}. Nothing changed.")
    return matches[0]


def option(argv, name):
    for index, arg in enumerate(argv):
        if arg == name and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def select_run(root, started, argv, explicit=None):
    directory = explicit or option(argv, "--output-dir")
    paths = [(root / directory).resolve() / "config.json"] if directory else list((root / "outputs").glob("*/config.json"))
    candidates = []
    for path in paths:
        try:
            config = json.loads(path.read_text())
            if path.stat().st_mtime < started - 2:
                continue
            if (root / config["output_dir"]).resolve() != path.parent.resolve():
                continue
            if not (path.parent / "metrics.jsonl").is_file():
                continue
            candidates.append(path.parent.resolve())
        except (OSError, KeyError, ValueError):
            continue
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely identify the active output directory ({len(candidates)} matches). Use --run-dir with the printed output path. Nothing changed.")
    return candidates[0]


def latest_checkpoint(run):
    candidates = []
    required = ["config.json", "metadata.json", "training_state.pt",
                "actor/adapter_model.safetensors", "critic/adapter_model.safetensors"]
    for path in run.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit() and all((path / name).is_file() for name in required):
            candidates.append((int(suffix), path))
    if not candidates:
        raise RuntimeError(f"No complete checkpoint found in {run}; refusing to resume")
    return max(candidates, key=lambda item: item[0])


def resume_command(python, root, checkpoint, output, target):
    return [python, "-u", str(root / "scripts/train.py"), "--resume-from", str(checkpoint),
            "--output-dir", str(output), "--total-updates", str(target),
            "--skip-resume-checks", "--skip-final-eval"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=150)
    parser.add_argument("--run-dir", help="Normally detected from the single active CLI process and config creation time")
    args = parser.parse_args()
    if sys.platform != "linux" or args.target <= 0:
        parser.error("Requires Linux and a positive target")
    root = Path(__file__).resolve().parents[1]
    pid, identity, started, argv = active_training(root)
    if Path(f"/proc/{pid}/cwd").resolve() != root:
        raise RuntimeError("Training cwd differs from project root; nothing changed")
    # Fail before stopping if the installed CLI lacks the required final-eval flag.
    help_text = subprocess.check_output([sys.executable, str(root / "scripts/train.py"), "--help"], text=True)
    if "--skip-final-eval" not in help_text:
        raise RuntimeError("Update scripts/train.py first; nothing changed")
    run = select_run(root, started, argv, args.run_dir)
    if process_start(pid)[0] != identity:
        raise RuntimeError("Training process changed; nothing changed")
    print(f"Active training PID: {pid}\nActive output: {run}\nRequested final update: {args.target}", flush=True)
    (run / "STOP_AFTER_STEP").touch()
    print("Requested save after the current update. Waiting for the old training process to exit...", flush=True)
    last_notice = time.monotonic()
    while True:
        try:
            if process_start(pid)[0] != identity:
                break
        except (FileNotFoundError, ProcessLookupError):
            break
        if time.monotonic() - last_notice >= 30:
            print("Still waiting for current update/save; no forced interruption.", flush=True)
            last_notice = time.monotonic()
        time.sleep(1)
    step, checkpoint = latest_checkpoint(run)
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if not records or records[-1]["step"] != step:
        raise RuntimeError("Latest metrics and saved checkpoint disagree; inspect the old training log before resuming")
    print(f"Saved update {step}: {checkpoint}", flush=True)
    if step >= args.target:
        target_checkpoint = run / f"checkpoint-{args.target:05d}"
        print(f"Already reached/passed {args.target}; no new training will start.", flush=True)
        print(f"Requested checkpoint: {target_checkpoint} (exists={target_checkpoint.is_dir()})", flush=True)
        return
    output = root / "outputs" / f"search_r1_until_{args.target:05d}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    print(f"Resuming at {step + 1}; automatic finish at {args.target}. Output: {output}", flush=True)
    result = subprocess.run(resume_command(sys.executable, root, checkpoint, output, args.target), cwd=root)
    if result.returncode:
        raise SystemExit(result.returncode)
    final_step, final_checkpoint = latest_checkpoint(output)
    if final_step != args.target:
        raise RuntimeError(f"Stopped at {final_step}, not {args.target}; inspect training output")
    print(f"DONE: saved update {args.target}: {final_checkpoint}. No final evaluation was started.", flush=True)


if __name__ == "__main__":
    main()
