"""Read-only audit of explicitly selected PPO runs. No model/GPU dependency."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


FIELDS = ("reward", "exact_match", "valid_answer", "actor_loss", "critic_loss",
          "approx_kl", "reference_kl_sampled", "action_tokens", "searches",
          "actor_grad_norm", "critic_grad_norm", "clip_fraction", "step_seconds",
          "search_seconds", "collect_seconds", "update_seconds")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    raise ValueError(f"Incomplete/invalid JSON: {path}:{number}; copy after the writer is idle") from None
    return rows


def describe(values):
    values = [x for x in values if isinstance(x, (float, int)) and math.isfinite(x)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def correlation(rows, left, right):
    pairs = [(r[left], r[right]) for r in rows if isinstance(r.get(left), (int, float))
             and isinstance(r.get(right), (int, float)) and math.isfinite(r[left]) and math.isfinite(r[right])]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    denominator = math.sqrt(sum((x-xm)**2 for x in xs) * sum((y-ym)**2 for y in ys))
    return sum((x-xm)*(y-ym) for x, y in pairs) / denominator if denominator else None


def analyze(run_dirs, until=150, batch_size=8):
    by_step, rollouts, hashes, conflicts = {}, {}, {}, []
    for directory in map(Path, run_dirs):
        for name in ("metrics.jsonl", "rollouts.jsonl", "warmup.jsonl", "config.json", "evaluation.jsonl"):
            path = directory / name
            if path.exists():
                hashes[str(path.resolve())] = sha(path)
        for row in read_rows(directory / "metrics.jsonl"):
            step = row.get("step")
            if not isinstance(step, int) or not 1 <= step <= until:
                continue
            if step in by_step and by_step[step] != row:
                conflicts.append(step)
            by_step[step] = row
        for row in read_rows(directory / "rollouts.jsonl"):
            step = row.get("step")
            if not isinstance(step, int) or not 1 <= step <= until:
                continue
            # Older first-only logs may not have batch_index: never pretend they contain all eight.
            key = (step, row.get("batch_index"), row.get("id"))
            if key in rollouts and rollouts[key] != row:
                raise ValueError(f"Conflicting rollout records at {key}")
            rollouts[key] = row
    if conflicts:
        raise ValueError(f"Conflicting metric steps {sorted(set(conflicts))}; select a single run lineage")
    rows = [by_step[key] for key in sorted(by_step)]
    windows = []
    for start in range(1, until + 1, 25):
        selected = [r for r in rows if start <= r["step"] < start + 25]
        windows.append({"start": start, "end": min(start+24, until), "observed_steps": len(selected),
                        **{field: describe([r.get(field) for r in selected]) for field in FIELDS}})
    response_lengths, search_lengths, trace_counts = [], [], Counter()
    for row in rollouts.values():
        trace_counts[row["step"]] += 1
        for event in row.get("events", []):
            tokens = event.get("generated_tokens")
            if tokens is not None and event.get("type") == "answer":
                response_lengths.append(tokens)
            if tokens is not None and event.get("type") == "search":
                search_lengths.append(tokens)
    for window in windows:
        traces = [r for r in rollouts.values() if window["start"] <= r["step"] <= window["end"]]
        window["answer_action_tokens"] = describe([e.get("generated_tokens") for r in traces
            for e in r.get("events", []) if e.get("type") == "answer"])
        window["search_count_distribution"] = dict(Counter(r.get("searches") for r in traces))
    finish = Counter()
    for row in rows:
        finish.update(row.get("finish_reasons", {}))
    return {"observed_steps": len(rows), "missing_steps": sorted(set(range(1, until+1)) - set(by_step)),
            "input_sha256": hashes, "windows": windows,
            "overall": {field: describe([r.get(field) for r in rows]) for field in FIELDS},
            "zero_reward_steps": sum(r.get("reward") == 0 for r in rows),
            "finish_reasons": dict(finish), "last_train_examples_seen": rows[-1].get("train_examples_seen") if rows else None,
            "action_tokens_per_trajectory": describe([r["action_tokens"] / batch_size for r in rows if "action_tokens" in r]),
            "answer_action_tokens_from_available_traces": describe(response_lengths),
            "search_action_tokens_from_available_traces": describe(search_lengths),
            "rollout_records": len(rollouts), "rollout_records_by_step": dict(sorted(trace_counts.items())),
            "search_count_distribution": dict(Counter(r.get("searches") for r in rollouts.values())),
            "two_search_records": [{k: r.get(k) for k in ("step", "id", "reward", "finish_reason", "events")}
                                   for r in rollouts.values() if r.get("searches", 0) >= 2],
            "trace_coverage_complete": all(trace_counts[s] == batch_size for s in range(1, until+1)),
            "step_level_correlations_not_causal": {name: correlation(rows, "reward", name)
                for name in ("action_tokens", "searches", "reference_kl_sampled")},
            "notes": ["reward excludes KL shaping; actor_loss near zero is not accuracy",
                      "approx_kl = PPO old/new policy; reference_kl_sampled = sampled log ratio vs frozen base, not exact KL",
                      "action_tokens is the whole batch's generated search/answer/invalid tokens, not final response length",
                      "answer lengths cover valid answer actions only and include tags; older traces may lack token counts",
                      "No saved action masks: retrospective training mask correctness cannot be proven from these logs",
                      "Training batches change each step; correlations and window trends are descriptive, not held-out improvement"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--checkpoint-config", required=True)
    parser.add_argument("--checkpoint-metadata")
    parser.add_argument("--until", type=int, default=150)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.checkpoint_config).read_text())
    result = analyze(args.runs, args.until, config["rollout_batch_size"])
    result["checkpoint_config"] = config
    result["checkpoint_config_sha256"] = sha(args.checkpoint_config)
    if args.checkpoint_metadata:
        result["checkpoint_metadata"] = json.loads(Path(args.checkpoint_metadata).read_text())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("observed_steps", "missing_steps", "overall", "finish_reasons",
                                            "answer_action_tokens_from_available_traces", "trace_coverage_complete")},
                     ensure_ascii=False, indent=2))
    print("Audit saved:", destination)


if __name__ == "__main__":
    main()
