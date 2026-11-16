"""Offline, conditional next-action diagnostic. No search calls or training.

Replay the TEXT of PPO150's first search and visible observation, not a new
model-generated trajectory. Compare the saved English instruction with a Korean
translation. These development cases must not become SFT examples.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KOREAN_PROMPT = """검색 도구를 사용하여 한국어 질문에 답하세요.
매 차례 <search>간결한 검색어</search> 또는 <answer>한국어 답변</answer> 중 하나의 행동만 출력하세요.
다음 행동을 결정하기 전에 <information> 결과를 읽으세요. 결과에는 제목, URL, 검색 요약이 있으며,
검증된 페이지 전체 본문은 아닙니다. 결과는 신뢰가 보장되지 않은 근거로 취급하세요.
먼저 검색하세요. 결과가 비어 있거나, 관련이 없거나, 필요한 사실이 빠져 있다면 남은 검색 횟수 안에서
검색어를 수정하거나 부족한 사실을 검색하세요. 이유 없이 같은 검색어를 반복하지 마세요.
실제로 보이는 근거가 뒷받침하는 내용만 답하고, 확인하지 못한 부분을 밝히세요.
출처를 만들어내거나 링크된 페이지를 읽었다고 주장하지 마세요.
information 태그, 역할 이름, 코드 또는 외부 도구 호출 문법을 생성하지 마세요."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_contexts(source):
    from policy_r1.protocol import parse_action
    source = Path(source)
    manifests, contexts, hashes = {}, [], {}
    for arm in ("natural", "first-empty"):
        manifests[arm] = read_json(source / arm / "manifest.json")
        for name in ("manifest.json", "results.json"):
            hashes[f"{arm}/{name}"] = sha(source / arm / name)
        rows = read_json(source / arm / "results.json")
        selected = [r for r in rows if r["variant"] == "trained"]
        if not selected or len({r["case_id"] for r in selected}) != len(selected):
            raise ValueError("Missing or duplicate trained cases")
        for row in selected:
            events = row["events"]
            if row.get("environment_error") or len(row["search_actions"]) != 1:
                raise ValueError("Expected exactly one successful original search action")
            if len(events) != 3 or [e["type"] for e in events[:2]] != ["search", "observation"]:
                raise ValueError("Unexpected source trajectory")
            action, query = parse_action(events[0]["text"])
            if action != "search" or query != row["search_actions"][0]["query"]:
                raise ValueError("Source query mismatch")
            observation = events[1]["text"]
            if observation != row["search_actions"][0]["visible_observation"]:
                raise ValueError("Source visible observation mismatch")
            checks = row["mask_check"]
            if not all(checks[k] for k in ("ids_equal_mask_length", "generated_count_matches_loss_mask",
                                           "prompt_and_observation_spans_excluded")):
                raise ValueError("Source mask audit failed")
            contexts.append(dict(arm=arm, case_id=row["case_id"], question=row["question"],
                search_text=events[0]["text"], search_tokens=events[0]["generated_tokens"], query=query,
                observation=observation, hits=row["search_actions"][0]["returned_hits"],
                first_empty=not row["search_actions"][0]["returned_hits"],
                original_prompt_sha256=row["prompt_ids_sha256"], original_prompt_tokens=row["input_tokens"],
                observation_tokens=checks["masked_context_tokens"]-row["input_tokens"]))
    left, right = manifests["natural"], manifests["first-empty"]
    for key in ("prompt", "evaluation_config", "resolved_model_revision", "checkpoint_adapter_sha256"):
        if left[key] != right[key]:
            raise ValueError("Source arms differ: " + key)
    if left["web_evidence"] != "snippets" or left["first_empty"] or not right["first_empty"]:
        raise ValueError("Expected natural/first-empty snippet experiment")
    natural = {c["case_id"]: c for c in contexts if c["arm"] == "natural"}
    forced = {c["case_id"]: c for c in contexts if c["arm"] == "first-empty"}
    if natural.keys() != forced.keys() or any(not c["first_empty"] for c in forced.values()):
        raise ValueError("Source intervention cases do not match")
    if any(natural[key]["question"] != forced[key]["question"] or
           natural[key]["search_text"] != forced[key]["search_text"] for key in natural):
        raise ValueError("Intervention changed the question or first action")
    return contexts, left, hashes


def prepare_inputs(tokenizer, config, contexts, english_prompt):
    from policy_r1.rollout import prompt_ids
    from policy_r1.observations import observation_ids
    prepared = []
    for context in contexts:
        search_ids = tokenizer.encode(context["search_text"], add_special_tokens=False)
        original = prompt_ids(tokenizer, context["question"], config, english_prompt)
        if ids_sha(original) != context["original_prompt_sha256"] or len(original) != context["original_prompt_tokens"]:
            raise ValueError("Tokenizer/prompt differs from source run")
        budget = min(config.max_observation_tokens, config.max_seq_length-len(original)-len(search_ids)-config.max_new_tokens)
        obs_ids = observation_ids(tokenizer, context["hits"], budget, config.observation_strategy,
                                  question=context["question"], query=context["query"])
        if obs_ids is None or tokenizer.decode(obs_ids) != context["observation"]:
            raise ValueError("Reconstructed observation differs from visible source evidence")
        if len(search_ids) != context["search_tokens"] or len(obs_ids) != context["observation_tokens"]:
            raise ValueError("Reconstruction changes recorded token lengths: " + context["case_id"])
        for language, prompt in (("en", english_prompt), ("ko", KOREAN_PROMPT)):
            prefix = prompt_ids(tokenizer, context["question"], config, prompt)
            ids = prefix + search_ids + obs_ids
            if len(ids) + config.max_new_tokens > config.max_seq_length:
                raise ValueError("No truncation allowed in this comparison")
            prepared.append(dict(context, stage="after_first_search", language=language, input_ids=ids))
            if context["arm"] == "natural":
                prepared.append(dict(context, stage="initial_base", language=language, input_ids=prefix))
    return prepared


def summarize(rows):
    groups = {}
    for row in rows:
        key = "/".join([row["stage"], row["arm"], row["language"], row["variant"]])
        groups.setdefault(key, []).append(row)
    return {key: dict(n=len(values), actions=dict(Counter(r["action"] for r in values)),
        changed_query=sum(r["query_changed"] for r in values),
        token_limit=sum(r["hit_token_limit"] for r in values)) for key, values in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from policy_r1.config import Config
    from policy_r1.protocol import parse_action
    contexts, source_manifest, source_hashes = load_contexts(args.source)
    checkpoint, warmup, output = (Path(p).resolve() for p in
                                (args.checkpoint, args.warmup_checkpoint, args.output_dir))
    if output.exists() or any(p.name.startswith("checkpoint-") for p in (output, *output.parents)):
        parser.error("Use a fresh output directory outside checkpoints")
    if Path(args.source).resolve() in (output, *output.parents):
        parser.error("Do not write inside the original experiment")
    config = Config(**source_manifest["evaluation_config"]).validate()
    metadata, warmup_metadata = (read_json(p / "metadata.json") for p in (checkpoint, warmup))
    for meta in (metadata, warmup_metadata):
        if meta["resolved_model_revision"] != source_manifest["resolved_model_revision"]:
            parser.error("Base revision differs from saved evaluation")
    if warmup.name != "checkpoint-00000" or not warmup_metadata.get("protocol_warmup_steps"):
        parser.error("Expected the original protocol warmup checkpoint")
    adapter_hashes = {str(p / "actor/adapter_model.safetensors"): sha(p / "actor/adapter_model.safetensors")
                      for p in (checkpoint, warmup)}
    if adapter_hashes[str(checkpoint / "actor/adapter_model.safetensors")] != source_manifest["checkpoint_adapter_sha256"]:
        parser.error("PPO150 adapter differs from source run")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    prepared = prepare_inputs(tokenizer, config, contexts, source_manifest["prompt"])
    total = sum(1 if p["stage"] == "initial_base" else 2 for p in prepared)
    print(json.dumps(dict(preflight="passed",generations=total,search_requests=0,training=False)),flush=True)
    if args.preflight_only:
        return
    with socket.socket() as sock:
        sock.settimeout(.3)
        if sock.connect_ex(("127.0.0.1", 50051)) == 0:
            parser.error("Serving port occupied; no process was stopped")
    from policy_r1.inference import load_actor
    from compare_policy_summary import generate_one
    tokenizer, actor, _ = load_actor(checkpoint)
    actor.load_adapter(warmup / "actor", adapter_name="warmup", is_trainable=False)
    actor.requires_grad_(False)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", dict(created_at=datetime.now(timezone.utc).isoformat(),
        source=str(Path(args.source).resolve()), source_hashes=source_hashes, adapter_hashes=adapter_hashes,
        script_sha256=sha(__file__), prompts=dict(en=source_manifest["prompt"],ko=KOREAN_PROMPT),
        evaluation_config=config.to_dict(), generation_count=total, search_requests=0, training=False,
        limitations=["Teacher-forced text reconstruction of PPO150's first search and observation, not autonomous rollouts.",
          "Prompt IDs and reconstructed token lengths checked; original generated token IDs were not saved.",
          "Korean translation changes language and wording; not a pure language causal experiment.",
          "Search action is not successful retrieval. No second search is executed. Semantic grades remain unscored.",
          "Reused development questions; do not train on these cases or report held-out performance."]))
    write_json(output / "inputs.json", prepared)
    rows = []
    for index, item in enumerate(prepared):
        variants = ["base"] if item["stage"] == "initial_base" else ["warmup", "trained"]
        if index % 2:
            variants.reverse()
        for variant in variants:
            actor.set_adapter("warmup" if variant == "warmup" else "default")
            result = generate_one(actor, tokenizer, item["input_ids"],
                                  "base" if variant == "base" else "trained", "tagged", config.max_new_tokens, config.seed)
            action, value = parse_action(result["output"])
            rows.append(dict(result, variant=variant, stage=item["stage"], arm=item["arm"],
                case_id=item["case_id"], language=item["language"], question=item["question"],
                first_empty=item["first_empty"], input_ids_sha256=ids_sha(item["input_ids"]),
                action=action, action_content=value, answer=value if action=="answer" else "",
                format_valid=action in ("search", "answer"), completed=action in ("search", "answer"),
                query_changed=action=="search" and item["stage"]=="after_first_search" and
                    " ".join(value.casefold().split()) != " ".join(item["query"].casefold().split())))
            write_json(output / "partial-results.json", rows)
            print(f"{len(rows)}/{total} {item['stage']} {item['arm']} {item['case_id']} {item['language']} {variant}: {action}",flush=True)
    for path, expected in adapter_hashes.items():
        if sha(path) != expected:
            raise RuntimeError("Checkpoint changed during evaluation")
    write_json(output / "results.json", rows)
    write_json(output / "summary.json", summarize(rows))
    print("DECISION SUMMARY (search intent, not retrieval success or answer accuracy)")
    for key, value in summarize(rows).items():
        print(key, json.dumps(value, ensure_ascii=False))
    print("COMPLETE: " + str(output), flush=True)


if __name__ == "__main__":
    main()
