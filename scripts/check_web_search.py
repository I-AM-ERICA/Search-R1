"""One real SerpAPI call, without loading any model. May consume API quota."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_r1.web_search import SerpAPIRetriever, SearchUnavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="소상공인 지원사업 공고 site:bizinfo.go.kr")
    parser.add_argument("--engine", choices=("google", "naver"), default="google")
    parser.add_argument("--timeout", type=float, default=45,
                        help="Socket timeout in seconds for this diagnostic call only (default: 45)")
    args = parser.parse_args()
    try:
        retriever = SerpAPIRetriever(cache_size=0, timeout=args.timeout, engine=args.engine, require_page_evidence=True)
        hits = retriever.search(args.query, 3)
    except (SearchUnavailable, ValueError) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps({"query": args.query, "count": len(hits), "diagnostics": retriever.last_diagnostics,
        "results": [
        {key: hit[key] for key in ("title", "url", "snippet", "evidence_kind")} for hit in hits]}, ensure_ascii=False, indent=2))
    if not hits:
        raise SystemExit("API call returned no usable organic results; try another query")


if __name__ == "__main__":
    main()
