"""Evaluate a checkpoint without updating weights; test is never used for PPO or warmup."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_r1.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Validation only: evaluate a fixed random subset; 0 evaluates the full split")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Use the same seed and validation data for checkpoint comparisons")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")
    if args.max_samples and args.split != "val":
        parser.error("Subset evaluation is for validation; final test evaluation stays full")
    checkpoint = Path(args.checkpoint)
    config = Config(**json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    config.output_dir = args.output_dir or f"outputs/evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    from policy_r1.trainer import PPOExperiment
    experiment = PPOExperiment(config, resume_from=checkpoint)
    if args.max_samples:
        experiment.eval_data = random.Random(args.sample_seed).sample(
            experiment.val_data, min(args.max_samples, len(experiment.val_data)))
        selection = {"checkpoint": str(checkpoint.resolve()), "split": "val", "sample_seed": args.sample_seed,
                     "ids": [row["id"] for row in experiment.eval_data]}
        (experiment.output / "evaluation_sample.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"고정 검증 {len(experiment.eval_data)}문항 평가 시작 (seed={args.sample_seed}); 학습은 하지 않습니다.", flush=True)
    print(experiment.evaluate(full=not bool(args.max_samples), split=args.split))
    print("평가 결과:", experiment.output.resolve())


if __name__ == "__main__":
    main()
