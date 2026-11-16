"""One real search-first policy answer, without starting a server or training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--region", default="서울")
    parser.add_argument("--category", default="음식점")
    parser.add_argument("--search-engine", choices=("google", "naver"), default="naver")
    parser.add_argument("--output", help="Write full diagnostics to a new JSON file")
    args = parser.parse_args()
    from policy_r1.runtime_env import load_runtime_env
    from policy_r1.web_search import SerpAPIRetriever
    from policy_r1.inference import load_actor
    from policy_r1.mvp import MVPPolicyInference
    load_runtime_env()
    output = Path(args.output) if args.output else None
    if output and output.exists():
        parser.error("Use a new output file")
    search = SerpAPIRetriever(engine=args.search_engine, timeout=20, require_page_evidence=True)
    tokenizer, actor, config = load_actor(args.checkpoint)
    result = MVPPolicyInference(tokenizer, actor, config, search).answer(
        args.question, region=args.region, category=args.category)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key:result[key] for key in ("answer", "sources", "search_requests", "evidence_mode")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
