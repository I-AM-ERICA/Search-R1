"""Download (only when requested), convert QA, and build the Wikipedia disk index."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_r1.prepare import prepare_qa
from policy_r1.sqlite_search import build_sqlite_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("qa", "index", "all"), default="all")
    parser.add_argument("--download", action="store_true", help="Download missing official source files")
    parser.add_argument("--qa-dir", default="data/search_r1/qa")
    parser.add_argument("--output-dir", default="data/search_r1/prepared")
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", default="data/search_r1/corpus/wiki-18.jsonl.gz")
    parser.add_argument("--index", default="data/search_r1/index/wiki18.sqlite")
    args = parser.parse_args()
    if args.stage in ("qa", "all"):
        qa = Path(args.qa_dir)
        for name in ("train.parquet", "test.parquet"):
            ensure_source(qa / name, "PeterJinGo/nq_hotpotqa_train", name, args.download)
        manifest = prepare_qa(qa / "train.parquet", qa / "test.parquet", args.output_dir,
                              args.validation_size, args.seed)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.stage in ("index", "all"):
        corpus = Path(args.corpus)
        ensure_source(corpus, "PeterJinGo/wiki-18-corpus", "wiki-18.jsonl.gz", args.download)
        print("Building disk BM25 index. Full Wikipedia may take hours and substantial disk space.", flush=True)
        print(json.dumps(build_sqlite_index(corpus, args.index), indent=2), flush=True)


def ensure_source(path, repository, filename, download):
    if path.is_file():
        return
    if not download:
        raise FileNotFoundError(f"{path}: 파일이 없습니다. --download로 내려받거나 경로를 지정하세요.")
    if path.name != filename:
        raise ValueError(f"Automatic download expects filename {filename}; provide the custom file yourself")
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=repository, repo_type="dataset", filename=filename, local_dir=str(path.parent))


if __name__ == "__main__":
    main()
