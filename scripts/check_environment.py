# 역할: 모델 다운로드 전에 GPU·라이브러리·정책 데이터와 검색 결과를 점검합니다.
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_r1.config import load_config
from policy_r1.data import load_datasets
from policy_r1.retrieval import build_retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/first_experiment.yaml")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--hardware-only", action="store_true", help="Skip data/index checks before preparation")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_only and args.hardware_only:
        parser.error("--data-only and --hardware-only are mutually exclusive")
    if not args.hardware_only:
        documents, train, val = load_datasets(config)
        print(f"documents={len(documents):,}, train={len(train):,}, val={len(val):,}")
        print(f"train_path={config.train_path}, val_path={config.val_path}, test_path={config.test_path}")
        retriever = build_retriever(config, documents)
        hits = retriever.search(train[0]["question"], config.top_k)
        print("검색 점검:", [(h["id"], round(h["score"], 3)) for h in hits])
        if hasattr(retriever, "close"):
            retriever.close()
    if args.data_only:
        return
    import torch
    import transformers
    import peft
    print(f"Python={sys.version.split()[0]}, torch={torch.__version__}, CUDA runtime={torch.version.cuda}")
    print(f"transformers={transformers.__version__}, peft={peft.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU 없음: GPU 서버의 커널인지, NVIDIA 드라이버가 설치되었는지 확인하세요.")
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}, free={free / 2**30:.1f}/{total / 2**30:.1f} GiB")
    if config.load_in_4bit:
        import bitsandbytes
        print("bitsandbytes=", bitsandbytes.__version__)
    print("점검 완료. 이 결과는 모델 로딩 및 학습의 성공을 보장하지 않습니다.")


if __name__ == "__main__":
    main()
