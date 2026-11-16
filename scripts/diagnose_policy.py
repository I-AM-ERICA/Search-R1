"""Run one Actor request and print search/observation diagnostics, without starting gRPC.

Output includes model-generated search queries. By default it omits answer text, credentials,
and full retrieved pages. Stop the serving process before loading this probe.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def probe(engine, question, request_timeout=85, show_answer=False, show_invalid_output=False,
          region="서울", category="음식점"):
    from policy_r1 import inference
    from policy_r1.web_search import SearchUnavailable
    from policy_r1.rollout import RolloutCancelled

    report = {"search_calls": [], "trajectory": None}
    original_search = engine.retriever
    original_rollout = inference.collect_rollout

    class TracedSearch:
        require_page_evidence = getattr(original_search, "require_page_evidence", False)

        def read_pages(self, urls, **kwargs):
            hits = original_search.read_pages(urls, **kwargs)
            report["previous_pages"] = [{key: hit.get(key) for key in ("id", "url", "evidence_kind")} for hit in hits]
            return hits

        def search(self, query, top_k, **kwargs):
            call = {"query": query, "top_k": top_k}
            report["search_calls"].append(call)
            started = time.monotonic()
            try:
                hits = original_search.search(query, top_k, **kwargs)
                call["returned_hits"] = [
                    {key: hit.get(key) for key in ("id", "title", "url", "evidence_kind")}
                    for hit in hits]
                call["diagnostics"] = dict(getattr(original_search, "last_diagnostics", {}))
                return hits
            except SearchUnavailable as error:
                call["error_code"] = error.code
                raise
            finally:
                call["seconds"] = round(time.monotonic() - started, 3)

    def traced_rollout(*args, **kwargs):
        trajectory = original_rollout(*args, **kwargs)
        observed = args[2] if len(args) > 2 else kwargs.get("retriever")
        report["source_selection"] = list(getattr(observed, "selection_log", []))
        report["evidence_focus"] = list(getattr(observed, "focus_log", []))
        events = []
        for event in trajectory.events:
            entry = {"type": event["type"]}
            if event["type"] == "observation":
                ids = event.get("document_ids", [])
                entry.update({"returned_ids": ids,
                    "visible_ids": [key for key in ids if f"[{key}]" in event["text"]],
                    "preview": event["text"][:240]})
            elif event["type"] == "search":
                entry["action"] = event["text"][:1200]
            else:
                entry["generated_tokens"] = event.get("generated_tokens")
                if event["type"] == "invalid":
                    text = event["text"]
                    entry.update(hit_token_limit=event.get("hit_token_limit"),
                                 model_turn_end=event.get("model_turn_end"),
                                 starts_with_answer=text.lstrip().startswith("<answer>"),
                                 has_answer_close="</answer>" in text)
                    if show_invalid_output:
                        # Diagnostic only. Never treat incomplete text as a served answer.
                        entry["output_head"] = text[:240]
                        entry["output_tail"] = text[-240:] if len(text) > 240 else ""
            events.append(entry)
        report["trajectory"] = {"finish_reason": trajectory.finish_reason,
                                "searches": trajectory.searches, "events": events}
        return trajectory

    engine.retriever = TracedSearch()
    try:
        with patch.object(inference, "collect_rollout", traced_rollout):
            result = engine.answer(question, region=region, category=category,
                                   deadline=time.monotonic() + request_timeout)
        report.update(status="success", source_count=len(result["sources"]),
                      search_requests=result.get("search_requests", len(report["search_calls"])))
        if show_answer:
            report["answer"] = result["answer"]
            report["sources"] = [{key: source[key] for key in ("title", "url")} for source in result["sources"]]
    except (inference.InferenceFailure, SearchUnavailable, RolloutCancelled, ValueError) as error:
        report.update(status="failed", error_type=type(error).__name__, reason=str(error))
    finally:
        engine.retriever = original_search
    report["limits"] = {name: getattr(engine.config, name) for name in (
        "max_prompt_tokens", "max_new_tokens", "max_observation_tokens", "max_seq_length", "max_searches")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question", default="소상공인시장진흥공단 정책자금의 공통 지원 자격을 공식 자료로 확인해줘.")
    parser.add_argument("--search-engine", choices=("google", "naver"), default="naver")
    parser.add_argument("--search-timeout", type=float, default=30)
    parser.add_argument("--show-answer", action="store_true", help="Also print the final answer and source URLs")
    parser.add_argument("--show-invalid-output", action="store_true", help="Print bounded head/tail of failed model output")
    parser.add_argument("--max-new-tokens", type=int, help="Serving-only limit; default: checkpoint setting")
    args = parser.parse_args()
    from policy_r1.runtime_env import load_runtime_env
    load_runtime_env()
    from policy_r1.inference import PolicyInference, load_actor, with_generation_limit
    from policy_r1.web_search import SerpAPIRetriever
    search = SerpAPIRetriever(engine=args.search_engine, timeout=args.search_timeout, require_page_evidence=True)
    tokenizer, actor, config = load_actor(args.checkpoint)
    config = with_generation_limit(config, args.max_new_tokens)
    result = probe(PolicyInference(tokenizer, actor, config, search), args.question,
                   show_answer=args.show_answer, show_invalid_output=args.show_invalid_output)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    for variable in ("SERPAPI_API_KEY", "AI_INTERNAL_KEY"):
        secret = os.environ.get(variable)
        if secret:
            output = output.replace(secret, "[REDACTED]")
    print(output)


if __name__ == "__main__":
    main()
