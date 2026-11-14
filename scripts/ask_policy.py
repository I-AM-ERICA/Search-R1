"""Authenticated local gRPC smoke test. Uses real model + search API quota."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grpc
from america.ai.v1 import common_pb2, policy_pb2, policy_pb2_grpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--question", required=True)
    parser.add_argument("--region", default="서울")
    parser.add_argument("--category", default="음식점")
    args = parser.parse_args()
    key = os.environ.get("AI_INTERNAL_KEY", "")
    if not key:
        parser.error("Set AI_INTERNAL_KEY; it must match the inference server")
    request = policy_pb2.PolicyQuestion(question=args.question, top_k=3,
        store=common_pb2.StoreContext(region=args.region, category=args.category))
    try:
        with grpc.insecure_channel(args.address) as channel:
            response = policy_pb2_grpc.PolicyAssistantStub(channel).Ask(
                request, metadata=(("x-internal-key", key),), timeout=90)
        print(json.dumps({"answer": response.answer, "sources": [
            {"title": source.title, "url": source.url, "snippet": source.snippet} for source in response.sources]},
            ensure_ascii=False, indent=2))
    except grpc.RpcError as error:
        raise SystemExit(f"{error.code().name}: {error.details()}") from None


if __name__ == "__main__":
    main()
