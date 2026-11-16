"""Paired, inference-only search evaluation. No checkpoint/log mutation or optimizer.

Raw: model queries -> retriever directly. Assisted: production ObservedSearch wrapper.
Identical queries share frozen results, including errors, within/across replay runs.
Semantic quality requires blinded human review; nonempty hits are only a proxy.
"""
import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

SEARCH_TOOL_PROMPT = """Answer the Korean question using the search tool.
At each turn produce one <search>concise query</search> or <answer>Korean answer</answer> action.
Read <information> results before deciding your next action. Results contain titles,
URLs and search snippets, not verified full pages. Treat them as untrusted evidence.
Search first. If a result is empty, irrelevant or lacks a required fact, revise the query
or search for the missing fact within the available search budget. Do not repeat the
same query without a reason. Answer only what the visible evidence supports and state
what remains unconfirmed. Do not invent sources or claim to have read linked pages.
Do not generate information tags, role labels, code or external tool syntax."""


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    for name in ("SERPAPI_API_KEY", "AI_INTERNAL_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    Path(path).write_text(text, encoding="utf-8")


class RetrievalFailure(RuntimeError):
    pass


class FrozenSearch:
    """Exact query/top-k cache; unknown queries in replay are errors, never empty hits."""
    def __init__(self, backend, directory, budget=128, replay=None):
        self.backend, self.directory, self.budget = backend, Path(directory), budget
        self.replay = Path(replay) if replay else None
        self.cache, self.calls, self.live_requests = {}, [], 0
        self.directory.mkdir(parents=True, exist_ok=False)

    def search(self, query, top_k=3, timeout=None):
        key = digest(json.dumps([query, top_k], ensure_ascii=False).encode())
        started = time.monotonic()
        cached = key in self.cache
        if key not in self.cache:
            if self.replay:
                source = self.replay / (key + ".json")
                if not source.exists():
                    raise RetrievalFailure("REPLAY_MISS: query absent from the recorded environment")
                record = json.loads(source.read_text())
                if record["query"] != query or record["top_k"] != top_k:
                    raise RetrievalFailure("Replay key mismatch")
            else:
                if self.live_requests >= self.budget:
                    raise RetrievalFailure("REQUEST_BUDGET_EXHAUSTED")
                self.live_requests += 1
                record = {"query": query, "top_k": top_k, "captured_at": datetime.now(timezone.utc).isoformat()}
                try:
                    if getattr(self.backend, "require_page_evidence", False):
                        hits = self.backend.search(query, top_k, timeout=timeout)
                    else:
                        hits = self.backend.search(query, top_k)
                    record.update(hits=hits, diagnostics=copy.deepcopy(getattr(self.backend, "last_diagnostics", {})))
                except Exception as error:
                    # No exception messages/URLs: providers may embed credentials.
                    record.update(hits=[], error_code=getattr(error, "code", type(error).__name__))
            self.cache[key] = record
            save(self.directory / (key + ".json"), record)
        record = copy.deepcopy(self.cache[key])
        self.calls.append({**record, "snapshot_key": key, "cache_hit": cached or bool(self.replay),
                           "seconds": round(time.monotonic() - started, 4)})
        if record.get("error_code"):
            raise RetrievalFailure(record["error_code"])
        return record["hits"]


class ActionTrace:
    """One entry per model-initiated search, not per wrapper retry/API call."""
    def __init__(self, retriever, frozen, first_empty=False):
        self.retriever, self.frozen, self.first_empty = retriever, frozen, first_empty
        self.actions = []

    def search(self, query, top_k):
        row = {"query": query, "requests": [], "returned_hits": []}
        self.actions.append(row)
        start = len(self.frozen.calls)
        try:
            if self.first_empty and len(self.actions) == 1:
                row["intervention"] = "forced_first_empty_without_provider_request"
                return []
            hits = self.retriever.search(query, top_k)
            row["returned_hits"] = hits
            return hits
        finally:
            row["requests"] = copy.deepcopy(self.frozen.calls[start:])


def mask_check(trajectory, tokenizer=None, prompt_tokens=None):
    """Reconstruct event spans using actual generated counts, never decode/re-encode actions."""
    generated = sum(e.get("generated_tokens", 0) for e in trajectory.events if e["type"] != "observation")
    result = {"ids_equal_mask_length": len(trajectory.input_ids) == len(trajectory.action_mask),
            "generated_tokens": generated, "loss_action_tokens": sum(trajectory.action_mask),
            "generated_count_matches_loss_mask": generated == sum(trajectory.action_mask),
            "masked_context_tokens": len(trajectory.action_mask) - sum(trajectory.action_mask)}
    if tokenizer is not None and prompt_tokens is not None:
        cursor = prompt_tokens
        valid = not any(trajectory.action_mask[:cursor])
        for event in trajectory.events:
            start = cursor
            if event["type"] == "observation":
                while cursor < len(trajectory.action_mask) and not trajectory.action_mask[cursor]:
                    cursor += 1
                valid = valid and tokenizer.decode(trajectory.input_ids[start:cursor]) == event["text"]
            else:
                cursor += event["generated_tokens"]
                valid = valid and all(trajectory.action_mask[start:cursor])
                valid = valid and tokenizer.decode(trajectory.input_ids[start:cursor], skip_special_tokens=True) == event["text"]
        result["prompt_and_observation_spans_excluded"] = bool(valid and cursor == len(trajectory.input_ids))
    return result


def run_one(actor, tokenizer, config, frozen, case, variant, pipeline, prompt, seed, first_empty=False):
    import torch
    from compare_policy_summary import adapter_variant
    from policy_r1.rollout import collect_rollout, prompt_ids
    from policy_r1.inference import ObservedSearch
    from policy_r1.serving_queries import fallback_search_query
    from policy_r1.protocol import score_answer

    question = case["question"]
    ids = prompt_ids(tokenizer, question, config, prompt)
    observed = ObservedSearch(frozen, lambda: False, float("inf"), max_requests=config.max_searches,
        official_only=True, tokenizer=tokenizer, observation_budget=config.max_observation_tokens,
        observation_strategy=config.observation_strategy,
        fallback_query=fallback_search_query(question)) if pipeline == "assisted" else frozen
    traced = ActionTrace(observed, frozen, first_empty)
    row = {"case_id": case["id"], "question": question, "variant": variant, "pipeline": pipeline,
           "prompt_ids_sha256": digest(json.dumps(ids).encode()), "input_tokens": len(ids),
           "intervention": "first_empty" if first_empty else "none"}
    torch.manual_seed(seed)
    started = time.monotonic()
    actor.set_adapter({"warmup": "warmup", "sft": "sft"}.get(variant, "default"))
    try:
        with adapter_variant(actor, "base" if variant == "base" else "trained"), torch.inference_mode():
            trajectory = collect_rollout(actor, tokenizer, traced, question, config, sample=False, system_prompt=prompt)
        row.update(answer=trajectory.answer, finish_reason=trajectory.finish_reason, events=trajectory.events,
                   searches=trajectory.searches, mask_check=mask_check(trajectory, tokenizer, len(ids)),
                   action_tokens=sum(trajectory.action_mask), format_complete=trajectory.finish_reason == "answer",
                   answer_tokens=len(tokenizer.encode(trajectory.answer, add_special_tokens=False)))
        observations = [e for e in trajectory.events if e["type"] == "observation"]
        for action, observation in zip(traced.actions, observations):
            action["visible_observation"] = observation["text"]
            action["visible_ids"] = [h["id"] for h in action["returned_hits"] if f"[{h['id']}]" in observation["text"]]
        if case.get("answers"):
            row["exact_match"] = score_answer(trajectory.answer, case["answers"], normalization=config.answer_normalization)["exact_match"]
    except RetrievalFailure as error:
        row.update(answer="", finish_reason="retrieval_error", environment_error=str(error), format_complete=False)
    finally:
        actor.set_adapter("default")
    row.update(search_actions=traced.actions, seconds=round(time.monotonic()-started, 3))
    row["nonempty_visible_search_proxy"] = any(a.get("visible_ids") for a in traced.actions)
    row["second_model_search"] = len(traced.actions) >= 2
    row["provider_requests"] = sum(len(a["requests"]) for a in traced.actions)
    row["selection_log"] = getattr(observed, "selection_log", [])
    return row


def rate(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "rate": sum(values)/len(values) if values else None}


def summarize(rows, reviews=None):
    reviews = {r["review_id"]: r for r in reviews or []}
    result = {}
    for variant in dict.fromkeys(r["variant"] for r in rows):
        selected = [r for r in rows if r["variant"] == variant]
        judged = [(r, reviews[r["review_id"]]) for r in selected if r["review_id"] in reviews]
        queries, success, completeness, accuracy, additional, recovery = [], [], [], [], [], []
        for row, review in judged:
            grades = review["search_actions"]
            for grade in grades:
                if grade.get("query_quality_0_to_2") is not None:
                    queries.append(grade["query_quality_0_to_2"] / 2)
            relevant = [g.get("relevant_visible_evidence") for g in grades]
            success.append((False if review.get("accuracy_0_to_2") is not None else None) if not grades else
                           True if any(v is True for v in relevant) else
                           False if all(v is False for v in relevant) else None)
            accuracy.append(None if review.get("accuracy_0_to_2") is None else review["accuracy_0_to_2"] == 2)
            completeness.append(None if review.get("coverage_0_to_2") is None else review["coverage_0_to_2"] == 2)
            # Conditional success only when a real second model search follows insufficient evidence.
            if len(grades) >= 2 and grades[0].get("sufficient_visible_evidence") is False:
                additional.append(grades[1].get("new_useful_evidence"))
            if grades and grades[0].get("sufficient_visible_evidence") is False:
                recovery.append(grades[1].get("new_useful_evidence") if len(grades) >= 2 else False)
        result[variant] = {"questions": len(selected),
            "environment_errors": sum("environment_error" in r for r in selected),
            "format_completion": rate([r["format_complete"] for r in selected]),
            "nonempty_search_PROXY": rate([r["nonempty_visible_search_proxy"] for r in selected]),
            "second_model_search_rate": rate([r["second_model_search"] for r in selected]),
            "exact_match_when_gold_available": rate([r.get("exact_match") for r in selected]),
            "human_relevant_search_success": rate(success),
            "human_query_quality_mean_normalized": rate(queries),
            "human_additional_search_success_given_first_insufficient": rate(additional),
            "human_recovery_over_all_first_insufficient_cases": rate(recovery),
            "human_fully_correct_answers": rate(accuracy), "human_complete_answers": rate(completeness)}
    return result


def make_reviews(rows, cases, seed):
    lookup = {c["id"]: c for c in cases}
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    reviews = []
    for index, row in enumerate(shuffled):
        row["review_id"] = f"R{index+1:04d}"
        reviews.append({"review_id": row["review_id"], "question": row["question"],
            "answer": row["answer"], "finish_reason": row["finish_reason"],
            "review_points": lookup[row["case_id"]].get("review_points", []),
            "reference_answers_for_review_only": lookup[row["case_id"]].get("answers"),
            "search_actions": [{**copy.deepcopy(a), "query_quality_0_to_2": None,
                "relevant_visible_evidence": None, "sufficient_visible_evidence": None,
                "new_useful_evidence": None, "notes": ""} for a in row["search_actions"]],
            "accuracy_0_to_2": None, "coverage_0_to_2": None, "unsupported_claims": None, "notes": ""})
    return reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup-checkpoint")
    parser.add_argument("--sft-checkpoint", help="Optional fresh-base SFT adapter; compare under its declared token/search environment")
    parser.add_argument("--environment", choices=("web", "wiki"), default="web")
    parser.add_argument("--web-evidence", choices=("pages", "snippets"), default="pages",
                        help="snippets disables ALL page fetching, including missing-snippet fallback")
    parser.add_argument("--prompt-mode", choices=("policy", "search_tool"), default="policy")
    parser.add_argument("--pipeline", choices=("raw", "assisted"), default="raw")
    default_cases = Path(__file__).with_name("search_cases.json")
    if not default_cases.exists():
        default_cases = Path(__file__).resolve().parents[1] / "evaluations/search_strategy/cases.json"
    parser.add_argument("--cases", default=str(default_cases))
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--max-observation-tokens", type=int)
    parser.add_argument("--observation-strategy", choices=("concat", "balanced_v1", "focused_v1"),
                        help="Explicit observation ablation; omitted preserves historical runs")
    parser.add_argument("--engine", choices=("google", "naver"), default="naver")
    parser.add_argument("--official-only", action="store_true",
                        help="Filter nonofficial hosts before spending page-fetch budget, same for every model")
    parser.add_argument("--search-timeout", type=float, default=30)
    parser.add_argument("--max-provider-requests", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay", help="Previous run directory; exact query misses fail closed")
    parser.add_argument("--first-result-empty", action="store_true", help="Separate robustness intervention, not natural search success")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    sys.path.append(str(repo / "scripts"))
    from policy_r1.inference import load_actor, POLICY_SYSTEM_PROMPT, with_generation_limit
    from policy_r1.data import load_qa
    from policy_r1.config import Config
    from policy_r1.rollout import prompt_ids
    from audit_ppo_run import sha

    checkpoint = Path(args.checkpoint).resolve()
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    saved_config = Config(**json.loads((checkpoint / "config.json").read_text()))
    destination = Path(args.output_dir).resolve()
    if destination.exists() or checkpoint == destination or checkpoint in destination.parents:
        parser.error("Use a new output directory outside all checkpoints")
    if args.max_cases < 1 or args.max_provider_requests < 1:
        parser.error("Limits must be positive")
    if args.pipeline == "assisted" and args.environment != "web":
        parser.error("Assisted wrapper is only for the web environment")
    if args.web_evidence == "snippets" and (args.environment != "web" or args.pipeline != "raw"):
        parser.error("Snippet-only behavior evaluation requires web/raw")
    if args.prompt_mode == "search_tool" and args.environment != "web":
        parser.error("search_tool prompt requires web environment")
    if args.first_result_empty and args.pipeline != "raw":
        parser.error("Run first-empty robustness only with raw model queries")
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(("127.0.0.1", 50051)) == 0:
            parser.error("Stop serve_policy.py before loading evaluation; no process was stopped")
    if args.environment == "wiki":
        val_path = repo / saved_config.val_path
        if sha(val_path) != metadata["data_identity"]["val"]:
            parser.error("Validation file differs from the training checkpoint")
        all_cases = load_qa(val_path)
        # Same deterministic 200-example set used by PPOExperiment.evaluate().
        cases = random.Random(saved_config.seed).sample(all_cases, min(saved_config.eval_max_samples or len(all_cases), len(all_cases)))[:args.max_cases]
        prompt = None
    else:
        cases = json.loads(Path(args.cases).read_text())["cases"][:args.max_cases]
        prompt = SEARCH_TOOL_PROMPT if args.prompt_mode == "search_tool" else POLICY_SYSTEM_PROMPT
    if len({c["id"] for c in cases}) != len(cases) or not cases:
        parser.error("Cases must be nonempty with unique IDs")
    tokenizer, actor, config = load_actor(checkpoint)
    for adapter in actor.peft_config.values():
        if adapter.bias != "none" or adapter.modules_to_save:
            parser.error("Requires LoRA-only checkpoint")
    config = with_generation_limit(config, args.max_new_tokens if args.max_new_tokens is not None else
                                   (512 if args.environment == "web" else None))
    if args.environment == "web":
        config = replace(config, max_observation_tokens=1200)
    if args.observation_strategy:
        config = replace(config, observation_strategy=args.observation_strategy)
    if args.max_seq_length is not None:
        config = replace(config, max_seq_length=args.max_seq_length)
    if args.max_observation_tokens is not None:
        config = replace(config, max_observation_tokens=args.max_observation_tokens)
    config.validate()
    variants = ["base", "trained"]
    warmup = None
    if args.warmup_checkpoint:
        warmup = Path(args.warmup_checkpoint).resolve()
        if warmup == destination or warmup in destination.parents:
            parser.error("Output cannot be within warmup checkpoint")
        wm = json.loads((warmup / "metadata.json").read_text())
        wc = json.loads((warmup / "config.json").read_text())
        if wm["resolved_model_revision"] != metadata["resolved_model_revision"] or wc["model_name"] != config.model_name:
            parser.error("Warmup and current checkpoint must use identical base weights")
        if not wm.get("protocol_warmup_steps") or warmup.name != "checkpoint-00000":
            parser.error("Warmup control must be checkpoint-00000 with recorded protocol_warmup_steps > 0")
        actor.load_adapter(str(warmup / "actor"), adapter_name="warmup", is_trainable=False)
        variants = ["base", "warmup", "trained"]
    sft = None
    if args.sft_checkpoint:
        if args.environment != "web" or args.pipeline != "raw":
            parser.error("SFT policy comparison requires web/raw environment")
        sft = Path(args.sft_checkpoint).resolve()
        if sft == destination or sft in destination.parents:
            parser.error("Output cannot be within SFT checkpoint")
        sm = json.loads((sft / "metadata.json").read_text())
        sc = Config(**json.loads((sft / "config.json").read_text())).validate()
        if (sm.get("checkpoint_kind") != "policy_search_sft"
                or sm["resolved_model_revision"] != metadata["resolved_model_revision"]
                or sc.model_name != config.model_name):
            parser.error("SFT must use identical base weights and recorded SFT metadata")
        from transformers import AutoTokenizer
        st = AutoTokenizer.from_pretrained(sft / "tokenizer", local_files_only=True)
        if (st.get_vocab() != tokenizer.get_vocab() or st.chat_template != tokenizer.chat_template
                or st.special_tokens_map != tokenizer.special_tokens_map):
            parser.error("SFT/base tokenizer mismatch")
        if sm.get("prompt_sha256") != digest(json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()):
            parser.error("SFT prompt differs from evaluation prompt")
        # Same environment for base/PPO/SFT, not each model's preferred limits.
        overrides = {name:getattr(sc,name) for name in ("max_seq_length","max_prompt_tokens",
            "max_new_tokens","max_observation_tokens","max_searches","top_k","observation_strategy")}
        for name in ("max_seq_length","max_new_tokens","max_observation_tokens","observation_strategy"):
            explicit=getattr(args,name)
            if explicit is not None and explicit!=overrides[name]:
                parser.error("Explicit evaluation override conflicts with SFT environment: "+name)
        config=replace(config,**overrides).validate()
        actor.load_adapter(str(sft / "actor"), adapter_name="sft", is_trainable=False)
        variants.append("sft")
    if config.max_seq_length > actor.config.max_position_embeddings:
        parser.error("Evaluation context limit exceeds base model capacity")
    for case in cases:
        prompt_ids(tokenizer, case["question"], config, prompt)
    destination.mkdir(parents=True, exist_ok=False)
    replay = Path(args.replay).resolve() if args.replay else None
    if replay:
        prior = json.loads((replay / "manifest.json").read_text())
        if (prior["environment"] != args.environment or prior["engine"] != args.engine
                or prior.get("official_only", False) != args.official_only
                or prior.get("web_evidence", "pages") != args.web_evidence):
            parser.error("Replay environment/engine mismatch")
    backend = None
    if args.environment == "wiki":
        from policy_r1.sqlite_search import SQLiteRetriever
        backend = SQLiteRetriever(repo / config.corpus_path)
        if backend.metadata != metadata["data_identity"]["index_metadata"]:
            parser.error("Wiki index identity mismatch")
    elif not replay:
        from policy_r1.web_search import SerpAPIRetriever
        from policy_r1.official_pages import fetch_public_page, PageUnavailable
        pages = {}
        def fetch_once(url, timeout):
            if url not in pages:
                try:
                    pages[url] = {"url": url, "text": fetch_public_page(url, timeout=timeout)}
                except PageUnavailable:
                    pages[url] = {"url": url, "error": "page_unavailable"}
                save(destination / "pages.json", pages)
            if "error" in pages[url]:
                raise PageUnavailable("page_unavailable")
            return pages[url]["text"]
        backend = SerpAPIRetriever(engine=args.engine, timeout=args.search_timeout, cache_size=0,
                                  require_page_evidence=args.web_evidence == "pages", page_fetcher=fetch_once,
                                  snippet_only=args.web_evidence == "snippets",
                                  official_only=args.official_only)
    frozen = FrozenSearch(backend, destination / "snapshots", args.max_provider_requests,
                          replay / "snapshots" if replay else None)
    manifest = {"environment": args.environment, "engine": args.engine, "pipeline": args.pipeline,
        "web_evidence": args.web_evidence, "prompt_mode": args.prompt_mode,
        "official_only": args.official_only,
        "checkpoint": str(checkpoint), "warmup_checkpoint": str(warmup) if warmup else None,
        "sft_checkpoint": str(sft) if sft else None,
        "sft_adapter_sha256": sha(sft / "actor/adapter_model.safetensors") if sft else None,
        "resolved_model_revision": metadata["resolved_model_revision"], "checkpoint_config": saved_config.to_dict(),
        "evaluation_config": config.to_dict(), "prompt": prompt or "checkpoint language prompt",
        "seed": args.seed, "do_sample": False, "first_empty": args.first_result_empty,
        "variants": variants, "case_count": len(cases), "checkpoint_adapter_sha256": sha(checkpoint / "actor/adapter_model.safetensors"),
        "code_hashes": {str(p.relative_to(repo)): sha(p) for p in (repo / "policy_r1").glob("*.py")},
        "created_at": datetime.now(timezone.utc).isoformat(), "notes": [
            "No answer/gold/review points supplied to model; model chooses all search actions",
            "Same-query shared snapshots freeze responses/errors; distinct live queries are not a globally frozen web index",
            "Assisted mode is ObservedSearch-only, not full HTTP/gRPC UI/rewrite pipeline",
            ("Snippet-only: provider order, no page fetch, no automatic query rewrite or retry" if args.web_evidence == "snippets"
             else "Web retrieval uses existing official-first ranking/body extraction even in raw mode"),
            "Quality fields require source-grounded human review; automatic hit/format rates are proxies"]}
    save(destination / "manifest.json", manifest)
    save(destination / "cases.json", cases)
    rows = []
    for index, case in enumerate(cases):
        # Rotate order to avoid always giving the first model cold retrieval latency.
        offset = index % len(variants)
        for variant in variants[offset:] + variants[:offset]:
            row = run_one(actor, tokenizer, config, frozen, case, variant, args.pipeline, prompt,
                          args.seed, args.first_result_empty)
            rows.append(row)
            save(destination / "partial-results.json", rows)
            print(f"{len(rows)}/{len(cases)*len(variants)} {case['id']} {variant}: {row['finish_reason']}", flush=True)
    reviews = make_reviews(rows, cases, args.seed)
    save(destination / "results.json", rows)
    save(destination / "review-blind.json", reviews)
    save(destination / "summary.json", summarize(rows))
    manifest["actual_unique_provider_requests"] = frozen.live_requests
    if sha(checkpoint / "actor/adapter_model.safetensors") != manifest["checkpoint_adapter_sha256"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    if sft and sha(sft / "actor/adapter_model.safetensors") != manifest["sft_adapter_sha256"]:
        raise RuntimeError("SFT checkpoint changed during evaluation")
    save(destination / "manifest.json", manifest)
    print("SEARCH COMPARISON COMPLETE:", destination)
    print("Semantic grades pending review-blind.json. No training was performed.")


if __name__ == "__main__":
    main()
