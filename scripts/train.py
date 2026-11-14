# 역할: train.ipynb와 동일한 PPO 학습을 터미널에서 실행하는 보조 진입점입니다.
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_r1.config import Config, load_config
from policy_r1.data import load_qa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Default: configs/search_r1.yaml; resume defaults to checkpoint config")
    parser.add_argument("--resume-from")
    parser.add_argument("--output-dir")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--epochs", type=int, help="Total data passes, including completed training")
    budget.add_argument("--total-updates", type=int)
    parser.add_argument("--warmup-steps", type=int, default=None, help="Fresh full run: 128; demo: 0; resume: skipped")
    parser.add_argument("--final-test", action="store_true", help="Run full held-out test only after final model selection")
    parser.add_argument("--skip-resume-checks", action="store_true",
                        help="Resume only: skip initial evaluation/rollout diagnostics after separate retrieval verification")
    parser.add_argument("--skip-final-eval", action="store_true",
                        help="Stop after training/checkpoint save; evaluate separately later")
    args = parser.parse_args()
    if args.skip_resume_checks and not args.resume_from:
        parser.error("--skip-resume-checks requires --resume-from")
    if args.resume_from and not args.config:
        config = Config(**json.loads((Path(args.resume_from) / "config.json").read_text()))
    else:
        config = load_config(args.config or "configs/search_r1.yaml")
    if args.epochs is not None:
        if args.epochs < 1 or config.train_sampling != "shuffle":
            parser.error("--epochs requires a positive integer and train_sampling: shuffle")
        config.total_updates = math.ceil(len(load_qa(config.train_path)) * args.epochs / config.rollout_batch_size)
    elif args.total_updates is not None:
        config.total_updates = args.total_updates
    config.output_dir = args.output_dir or f"outputs/search_r1_{datetime.now():%Y%m%d_%H%M%S_%f}"
    config.validate()
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else (0 if config.allow_demo else 128)
    if warmup_steps < 0:
        parser.error("--warmup-steps must be nonnegative")
    from policy_r1.trainer import PPOExperiment
    experiment = PPOExperiment(config, resume_from=args.resume_from)
    if experiment.step >= config.total_updates:
        parser.error("total_updates must exceed checkpoint step; use --epochs or --total-updates")
    print("저장 경로:", experiment.output.resolve())
    print("목표 PPO updates:", config.total_updates)
    if not args.skip_resume_checks:
        print("초기 평가 (고정 검증 subset):", experiment.evaluate())
    if not args.resume_from and warmup_steps:
        experiment.warmup_protocol(steps=warmup_steps, learning_rate=1e-4)
        experiment.save()
        print("형식 SFT 후 평가:", experiment.evaluate())
    if not args.skip_resume_checks:
        report = experiment.diagnose_rollouts(count=8, sample=True)
        print("PPO 사전 점검:", report["summary"])
        if report["summary"]["valid_answer"] == 0:
            raise RuntimeError("유효한 답변이 없습니다. diagnostics.jsonl의 검색/형식/길이와 warmup 설정을 확인하세요.")
    experiment.train()
    if (experiment.output / "STOP_AFTER_STEP").exists():
        print("Saved and stopped by request; skipping final evaluation.")
        return
    if args.skip_final_eval:
        print(f"Training finished at update {experiment.step}; checkpoint saved. Final evaluation skipped.")
        return
    print("학습 후 평가 (동일 subset):", experiment.evaluate())
    print("전체 검증 평가:", experiment.evaluate(full=True))
    if args.final_test:
        print("최종 test 평가:", experiment.evaluate(split="test"))


if __name__ == "__main__":
    main()
