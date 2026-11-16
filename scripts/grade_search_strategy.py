"""Aggregate explicit human judgements without treating search hits as correct answers."""
import argparse
import json
import math
from pathlib import Path
import random

from compare_search_strategy import summarize, save


def validate(rows, reviews):
    expected = {r["review_id"]: r for r in rows}
    if len({r["review_id"] for r in reviews}) != len(reviews):
        raise ValueError("Duplicate review IDs")
    for review in reviews:
        row = expected[review["review_id"]]
        if review["question"] != row["question"] or review["answer"] != row["answer"]:
            raise ValueError("Do not modify questions/answers in the review file")
        if len(review["search_actions"]) != len(row["search_actions"]):
            raise ValueError("Search action count differs from recorded run")
        for grade in [review] + review["search_actions"]:
            for key in ("accuracy_0_to_2", "coverage_0_to_2", "query_quality_0_to_2"):
                value = grade.get(key)
                if value is not None and (type(value) is not int or value not in (0, 1, 2)):
                    raise ValueError(f"Invalid grade {key}")
            for key in ("unsupported_claims", "relevant_visible_evidence", "sufficient_visible_evidence", "new_useful_evidence"):
                if grade.get(key) is not None and type(grade[key]) is not bool:
                    raise ValueError(f"Expected boolean or null: {key}")


def paired_stats(pairs):
    if not pairs:
        return {"n": 0}
    diffs = [b-a for a,b in pairs]
    wins, losses = sum(d>0 for d in diffs), sum(d<0 for d in diffs)
    discordant = wins + losses
    # Exact paired binary test. Exploratory only, no claim of representative sampling.
    p = min(1.0, 2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.0
    rng = random.Random(42)
    bootstrap = sorted(sum(rng.choice(diffs) for _ in diffs)/len(diffs) for _ in range(2000))
    return {"n": len(pairs), "right_minus_left": sum(diffs)/len(diffs), "wins": wins, "losses": losses,
            "ties": len(pairs)-discordant, "paired_bootstrap_95_percent": [bootstrap[49], bootstrap[1949]] if discordant else None,
            "bootstrap_note": "All observed paired outcomes are identical; a degenerate [0,0] bootstrap is not an equivalence bound" if not discordant else "paired resampling, 2000 replicates",
            "exact_mcnemar_p": p, "note": "Exploratory, multiple comparisons unadjusted; a small/nonrandom set cannot establish generalization"}


def aggregate(rows, reviews):
    validate(rows, reviews)
    judged = {r["review_id"]: r for r in reviews}
    result = {"by_variant": summarize(rows, reviews), "paired": {},
              "reviewer_types": sorted({r.get("reviewer_type", "unspecified") for r in reviews}),
              "review_note": "human_* keys identify review-based metrics; reviewer_types identifies who actually graded them"}
    for left in ("base", "warmup"):
        for metric in ("exact_match", "fully_correct", "content_complete"):
            groups = {}
            for row in rows:
                grade = judged.get(row["review_id"], {})
                value = row.get("exact_match") if metric == "exact_match" else grade.get(
                    "accuracy_0_to_2" if metric == "fully_correct" else "coverage_0_to_2")
                if value is not None:
                    groups.setdefault(row["case_id"], {})[row["variant"]] = int(value == (1 if metric == "exact_match" else 2))
            pairs = [(g[left],g["trained"]) for g in groups.values() if left in g and "trained" in g]
            result["paired"][left+"_vs_trained_"+metric] = paired_stats(pairs)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Choose a new output file")
    rows = json.loads((Path(args.run)/"results.json").read_text())
    reviews = json.loads(Path(args.reviews).read_text())
    result = aggregate(rows, reviews)
    save(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
