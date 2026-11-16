"""Evaluate a checkpoint without updating weights; test is never used for PPO or warmup."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_r1.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    config = Config(**json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
    config.output_dir = args.output_dir or f"outputs/evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    from policy_r1.trainer import PPOExperiment
    experiment = PPOExperiment(config, resume_from=checkpoint)
    print(experiment.evaluate(full=True, split=args.split))
    print("평가 결과:", experiment.output.resolve())


if __name__ == "__main__":
    main()
