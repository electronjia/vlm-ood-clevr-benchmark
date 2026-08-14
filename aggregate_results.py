"""
aggregate_results.py — Aggregate BLIP-2 baseline predictions into summary tables.

Extracted from the exploratory notebook. Groups the per-question results into the
overall / by-question-type / by-complexity / by-type-complexity tables used in
the report. Optionally does the same aggregation for the Q-Former metrics
(mean_sim, eff_rank) if that CSV is provided.

Usage:
    python aggregate_results.py --results blip2_clevr_results_final.csv
    python aggregate_results.py --results blip2_clevr_results_final.csv \
        --qformer qformer_metrics_all_questions.csv
"""

import argparse
import os

import pandas as pd


def aggregate_accuracy(df, out_dir):
    """Overall + by-type + by-complexity + by-type-complexity accuracy tables."""
    df["correct"] = df["correct"].astype(bool)

    # 1. Overall
    overall = pd.DataFrame({
        "level": ["overall"],
        "n_questions": [len(df)],
        "n_correct": [int(df["correct"].sum())],
        "accuracy": [df["correct"].mean()],
        "accuracy_percent": [df["correct"].mean() * 100],
    })
    overall.to_csv(os.path.join(out_dir, "results_overall.csv"), index=False)

    # 2. By question type
    by_type = (df.groupby("question_type")
                 .agg(n_questions=("correct", "count"),
                      n_correct=("correct", "sum"),
                      accuracy=("correct", "mean"))
                 .reset_index())
    by_type["accuracy_percent"] = by_type["accuracy"] * 100
    by_type = by_type.sort_values("accuracy_percent", ascending=False)
    by_type.to_csv(os.path.join(out_dir, "results_by_question_type.csv"), index=False)

    # 3. By complexity
    by_complexity = (df.groupby("question_complexity")
                       .agg(n_questions=("correct", "count"),
                            n_correct=("correct", "sum"),
                            accuracy=("correct", "mean"))
                       .reset_index())
    by_complexity["accuracy_percent"] = by_complexity["accuracy"] * 100
    by_complexity.to_csv(os.path.join(out_dir, "results_by_complexity.csv"), index=False)

    # 4. By type x complexity
    by_tc = (df.groupby(["question_type", "question_complexity"])
               .agg(n_questions=("correct", "count"),
                    n_correct=("correct", "sum"),
                    accuracy=("correct", "mean"))
               .reset_index())
    by_tc["accuracy_percent"] = by_tc["accuracy"] * 100
    by_tc.to_csv(os.path.join(out_dir, "results_by_type_complexity.csv"), index=False)

    print("Accuracy tables:")
    print(f"  overall: {overall['accuracy_percent'].iloc[0]:.2f}%  ({len(df)} questions)")
    print(f"  by_question_type: {len(by_type)} types")
    print(f"  by_complexity: {len(by_complexity)} levels")
    print(f"  by_type_complexity: {len(by_tc)} rows")


def aggregate_qformer(df, out_dir):
    """Overall + by-type + by-correct + by-complexity tables for mean_sim / eff_rank."""
    df["mean_sim"] = pd.to_numeric(df["mean_sim"], errors="coerce")
    df["eff_rank"] = pd.to_numeric(df["eff_rank"], errors="coerce")
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(bool)

    def agg(g):
        return pd.Series({
            "n_questions": g["mean_sim"].count(),
            "mean_sim_mean": g["mean_sim"].mean(),
            "mean_sim_std": g["mean_sim"].std(),
            "eff_rank_mean": g["eff_rank"].mean(),
            "eff_rank_std": g["eff_rank"].std(),
            "eff_rank_min": g["eff_rank"].min(),
            "eff_rank_max": g["eff_rank"].max(),
        })

    overall = pd.DataFrame([{
        "level": "overall",
        "n_questions": len(df),
        "mean_sim_mean": df["mean_sim"].mean(),
        "mean_sim_std": df["mean_sim"].std(),
        "eff_rank_mean": df["eff_rank"].mean(),
        "eff_rank_std": df["eff_rank"].std(),
        "eff_rank_min": df["eff_rank"].min(),
        "eff_rank_max": df["eff_rank"].max(),
    }])
    overall.to_csv(os.path.join(out_dir, "qformer_overall.csv"), index=False)

    df.groupby("question_type").apply(agg).reset_index() \
        .to_csv(os.path.join(out_dir, "qformer_by_question_type.csv"), index=False)
    if "correct" in df.columns:
        df.groupby("correct").apply(agg).reset_index() \
            .to_csv(os.path.join(out_dir, "qformer_by_correct.csv"), index=False)
    if "question_complexity" in df.columns:
        df.groupby("question_complexity").apply(agg).reset_index() \
            .to_csv(os.path.join(out_dir, "qformer_by_complexity.csv"), index=False)

    print("\nQ-Former metric tables:")
    print(f"  overall eff_rank: {overall['eff_rank_mean'].iloc[0]:.3f} "
          f"± {overall['eff_rank_std'].iloc[0]:.3f}")
    print(f"  overall mean_sim: {overall['mean_sim_mean'].iloc[0]:.3f} "
          f"± {overall['mean_sim_std'].iloc[0]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="BLIP-2 baseline results CSV")
    ap.add_argument("--qformer", default=None,
                    help="Optional Q-Former metrics CSV (mean_sim, eff_rank per question)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Reading {args.results}")
    aggregate_accuracy(pd.read_csv(args.results), args.out)

    if args.qformer:
        print(f"\nReading {args.qformer}")
        aggregate_qformer(pd.read_csv(args.qformer), args.out)

    print(f"\nAll tables written to {args.out}/")


if __name__ == "__main__":
    main()
