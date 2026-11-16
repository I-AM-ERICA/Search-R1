"""Live, held-out Korean policy evaluation; never trains or overwrites checkpoints.

Freezes the first fetched result per query within this run for fair comparisons.
Records raw agent behavior separately from the guarded MVP answer.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = [
    {"id": "mvp_accessibility", "region": "서울", "category": "음식점",
     "question": "2026년 서울 음식점의 장애인 편의시설 설치비 지원이 있는지, 대상과 제외 조건, 신청 기간 및 신청 방법을 공식 공고로 확인해줘."},
    {"id": "mvp_yellowumbrella", "region": "서울", "category": "소매업",
     "question": "2026년 서울 노란우산 희망장려금은 어떤 소상공인이 신청할 수 있고 제외 조건과 신청 절차는 무엇인지 알려줘."},
    {"id": "mvp_funding", "region": "서울", "category": "음식점",
     "question": "2026년 서울시 중소기업육성자금 중 소상공인이 이용할 수 있는 자금의 대상과 접수 기간, 신청 방법을 확인해줘."},
    {"id": "mvp_insufficient", "region": "서울", "category": "음식점",
     "question": "2026년 서울의 모든 음식점에 조건 없이 9억원을 지급하는 지원사업이 실제로 있는지 공식 공고로 확인해줘. 확인되지 않으면 그렇게 말해줘."},
]


def checkpoint_paths():
    sft = ROOT / "outputs/policy-sft-lr1e4-20261116T064528Z-436751/training"
    return {"base": sft/"checkpoint-00012",
            "ppo150": ROOT/"outputs/search_r1_until_00150_20261114_125545_672828/checkpoint-00150",
            **{f"sft_epoch{i}": sft/f"checkpoint-{i*6:05d}" for i in (1, 2, 3)}}


def normalize(text):
    return re.sub(r"[^a-z0-9가-힣]", "", text.lower())


def isolation_check(questions):
    # Exclude all known reviewed policy splits, not just the newest train split.
    files = sorted((ROOT/"outputs").glob("**/data/reviewed-records.json"))
    known = []
    for path in files:
        rows = json.loads(path.read_text())
        known.extend((r.get("question", ""), str(path), r.get("split")) for r in rows)
    if not known:
        raise ValueError("No policy training records found for isolation checks")
    checks = []
    for case in questions:
        q = normalize(case["question"])
        nearest = max(known, key=lambda r: SequenceMatcher(None, q, normalize(r[0])).ratio())
        similarity = SequenceMatcher(None, q, normalize(nearest[0])).ratio()
        if similarity >= .8:
            raise ValueError("Evaluation question overlaps known policy records: " + case["id"])
        checks.append(dict(id=case["id"], nearest_question=nearest[0], similarity=similarity))
    return dict(checked_files=[str(p) for p in files], questions=checks,
                note="Exact/character similarity checks; semantic policy-family leakage still requires review. No training is performed.")


class FrozenSearch:
    require_page_evidence = True

    def __init__(self, live, directory):
        self.live, self.directory, self.results = live, directory, {}
        self.failures = {}
        self.quota_exhausted = False
        self.last_diagnostics = {}

    def search(self, query, top_k, **kwargs):
        from policy_r1.web_search import SearchUnavailable
        key = (query, top_k)
        if key in self.failures:
            raise self.failures[key]
        if key not in self.results:
            if self.quota_exhausted:
                raise SearchUnavailable("Search quota unavailable; further live requests stopped", code="HTTP_429")
            try:
                hits = self.live.search(query, top_k, **kwargs)
            except SearchUnavailable as error:
                self.failures[key] = error
                self.quota_exhausted = self.quota_exhausted or error.code == "HTTP_429"
                raise
            self.results[key] = (hits, dict(self.live.last_diagnostics))
            name = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
            (self.directory/(name+".json")).write_text(json.dumps(dict(
                query=query, top_k=top_k, hits=hits, diagnostics=self.live.last_diagnostics), ensure_ascii=False, indent=2))
        hits, diagnostics = self.results[key]
        self.last_diagnostics = {**diagnostics, "evaluation_replay": True}
        return [dict(h) for h in hits]

    def read_pages(self, *args, **kwargs):
        return self.live.read_pages(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--search-engine", choices=("google", "naver"), default="naver")
    parser.add_argument("--cases", help="Optional JSON list with id, question, region, category")
    parser.add_argument("--guarded-only", action="store_true", help="Compare the current guarded pipeline without rerunning raw agent probes")
    args = parser.parse_args()
    from policy_r1.runtime_env import load_runtime_env
    load_runtime_env()
    out = Path(args.output_dir).resolve()
    if out.exists():
        parser.error("Use a new output directory")
    questions = json.loads(Path(args.cases).read_text()) if args.cases else QUESTIONS
    if not questions or len({q['id'] for q in questions}) != len(questions):
        parser.error("Use nonempty uniquely identified cases")
    paths = checkpoint_paths()
    for path in paths.values():
        for name in ("actor/adapter_model.safetensors", "metadata.json", "config.json", "tokenizer/tokenizer_config.json"):
            if not (path/name).is_file():
                parser.error("Missing checkpoint file: " + str(path/name))
    isolation = isolation_check(questions)
    out.mkdir(parents=True)
    manifest = dict(checkpoints={k:str(v) for k,v in paths.items()}, questions=questions,
        isolation=isolation, search_engine=args.search_engine, training=False,
        created_at=time.time(), status="preflight_only" if args.preflight_only else "started",
        model_selection=None, note="Do not select a model from format validity alone. Review policy relevance, dates, exclusions and amounts against saved official bodies.")
    manifest["code_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT/'policy_r1/mvp.py', ROOT/'policy_r1/inference.py', ROOT/'policy_r1/rollout.py']}
    manifest["guarded_only"] = args.guarded_only
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if args.preflight_only:
        print("PREFLIGHT COMPLETE: five variants; isolated questions; no GPU/search calls", out)
        return
    if not os.environ.get("SERPAPI_API_KEY", "").strip():
        parser.error("Set SERPAPI_API_KEY. Preflight saved; no model evaluation or live search was performed.")
    from policy_r1.inference import load_actor, PolicyInference
    from policy_r1.mvp import MVPPolicyInference
    from policy_r1.web_search import SerpAPIRetriever
    from scripts.diagnose_policy import probe
    tokenizer, actor, cfg = load_actor(paths["sft_epoch2"])
    cfg = replace(cfg, max_new_tokens=512, max_seq_length=4096, max_observation_tokens=1400,
                  max_prompt_tokens=768, max_searches=2, prompt_language="ko")
    snapshots = out/"search-snapshots"
    snapshots.mkdir()
    search = FrozenSearch(SerpAPIRetriever(engine=args.search_engine, timeout=20,
                           cache_size=0, require_page_evidence=True), snapshots)
    rows = []
    with (out/"results.jsonl").open("x") as stream:
        for label, path in paths.items():
            if label != "base":
                actor.load_adapter(path/"actor", adapter_name=label, is_trainable=False)
                actor.set_adapter(label)
            with actor.disable_adapter() if label == "base" else nullcontext():
                for case in questions:
                    row = dict(variant=label, case=case)
                    # Raw model-driven search/answer, with its failures intact.
                    row["agent"] = dict(status="not_run") if args.guarded_only else probe(PolicyInference(tokenizer, actor, cfg, search), case["question"],
                                         request_timeout=120, show_answer=True, show_invalid_output=True,
                                         region=case["region"], category=case["category"])
                    try:
                        row["mvp"] = MVPPolicyInference(tokenizer, actor, cfg, search).answer(
                            case["question"], region=case["region"], category=case["category"],
                            deadline=time.monotonic()+120)
                    except Exception as error:
                        row["mvp"] = dict(status="failed", error_type=type(error).__name__,
                                          error_code=getattr(error, "code", None))
                    rows.append(row)
                    stream.write(json.dumps(row,ensure_ascii=False)+"\n");stream.flush()
                    print(label, case["id"], row["agent"]["status"], flush=True)
    manifest["search_failures"] = sum(r["mvp"].get("error_type") == "SearchUnavailable" for r in rows)
    manifest["status"] = "partial_search_failures" if manifest["search_failures"] else "awaiting_semantic_review"
    manifest["completed_cases"] = len(rows)
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print("COMPARISON FINISHED; status="+manifest["status"]+"; review raw answers and official bodies before model selection:", out)


if __name__ == "__main__":
    main()
