"""Linux only: interrupt a legacy notebook ONLY while its main thread is searching.

No code is injected into the kernel. A stopped-process snapshot closes the race
between observing a safe collection phase and delivering SIGINT. The recovery
cell independently verifies the resulting Python traceback before saving.
"""
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def safe_collection(stack, pid):
    match = re.search(rf'^Thread {pid} \([^\n]*\): "MainThread"\n(.*?)(?=^Thread |\Z)',
                      stack, re.MULTILINE | re.DOTALL)
    if not match:
        return False
    main = match.group(1)
    return ("SQLiteRetriever.search (" in main and "sqlite_search.py:" in main
            and "PPOExperiment._collect_batch (" in main
            and "PPOExperiment.train (" in main
            and "PPOExperiment._update (" not in main)


def start_time(pid):
    # /proc stat's comm field can contain spaces and parentheses.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--py-spy", default="/tmp/search-r1-profiler/bin/py-spy")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if sys.platform != "linux" or args.pid <= 1:
        parser.error("This helper requires a Linux notebook PID > 1")
    command = Path(f"/proc/{args.pid}/cmdline").read_bytes()
    if b"ipykernel_launcher" not in command:
        parser.error("Target must be an ipykernel notebook process")
    identity = start_time(args.pid)

    def snapshot():
        return subprocess.run([args.py_spy, "dump", "--nonblocking", "--pid", str(args.pid)],
                              check=True, text=True, capture_output=True, timeout=5).stdout

    deadline = time.monotonic() + args.timeout
    print("Waiting for SQLite search inside PPO rollout collection...", flush=True)
    while time.monotonic() < deadline:
        if start_time(args.pid) != identity:
            raise RuntimeError("Process identity changed; no signal sent")
        if not safe_collection(snapshot(), args.pid):
            time.sleep(2)
            continue
        # Independent watchdog resumes the kernel if this helper is killed.
        watchdog_code = (
            "import os,sys,time,signal,pathlib; time.sleep(10); "
            "p=int(sys.argv[1]); f=pathlib.Path(f'/proc/{p}/stat'); "
            "same=f.exists() and f.read_text().rsplit(')',1)[1].split()[19]==sys.argv[2]; "
            "same and os.kill(p,signal.SIGCONT)"
        )
        watchdog = subprocess.Popen([sys.executable, "-c", watchdog_code, str(args.pid), identity],
                                    start_new_session=True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        paused = False
        try:
            os.kill(args.pid, signal.SIGSTOP)
            paused = True
            for _ in range(100):
                if "\nState:\tT" in Path(f"/proc/{args.pid}/status").read_text():
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError("Could not verify the process was stopped")
            # py-spy must not itself resume/lock the externally stopped process.
            stack = snapshot()
            if safe_collection(stack, args.pid):
                os.kill(args.pid, signal.SIGINT)
                print("SIGINT queued while stopped inside SQLite collection.", flush=True)
                print("Wait for KeyboardInterrupt in the notebook, then run the recovery cell.", flush=True)
                return
        finally:
            if paused:
                os.kill(args.pid, signal.SIGCONT)
            watchdog.terminate()
            watchdog.wait()
        time.sleep(2)
    raise SystemExit("No verified safe search window found; training was not interrupted.")


if __name__ == "__main__":
    main()
