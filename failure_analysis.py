"""
failure_analysis.py — Explain BLIP-2's CLEVR failures at the behavioral level.

Two zero-GPU analyses on the prediction CSV:
  1. Count confusion matrix + prediction-vs-truth distributions
     -> shows the model collapses to a default answer instead of counting
  2. Complexity paradox breakdown
     -> tests whether "complex > simple" accuracy is a question-type confound

Usage:
    python failure_analysis.py --csv qformer_metrics_all_questions.csv --out figures/
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_pred(p):
    """The prediction field echoes the prompt: 'Question: ... Answer: X'.
    Pull the text after the last 'Answer:'."""
    s = str(p)
    if "Answer:" in s:
        return s.split("Answer:")[-1].strip()
    return s.strip()


WORD_TO_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
}

def parse_int(x):
    """Extract the count a prediction expresses, handling BLIP-2's phrasings:
       '3' -> 3, 'There are two' -> 2, 'There are no gray balls' -> 0,
       'None' / 'as many as there are cubes' -> None (genuine non-answer)."""
    s = str(x).strip().lower()

    # explicit non-answers
    if s in ('none', 'none.', 'nan', ''):
        return None

    # "there are no X" / "are no X" means zero
    if re.search(r'\bare no\b', s):
        return 0

    # vague deflections are NOT numeric answers
    if 'as many' in s or 'a lot' in s:
        return None

    # a literal digit wins
    m = re.search(r'\d+', s)
    if m:
        return int(m.group())

    # otherwise a spelled-out number word
    for word, num in WORD_TO_NUM.items():
        if re.search(rf'\b{word}\b', s):
            return num

    return None


# Analysis 1: count question type confusion + distributions

def count_analysis(df, out_dir):
    count = df[df["question_type"] == "count"].copy()
    count["pred_clean"] = count["prediction"].apply(extract_pred)
    count["pred_int"] = count["pred_clean"].apply(parse_int)
    count["true_int"] = count["answer"].apply(parse_int)

    n = len(count)
    n_pred_none = count["pred_int"].isna().sum()
    print(f"\n{'='*55}")
    print("  COUNT FAILURE ANALYSIS")
    print(f"{'='*55}")
    print(f"  Total count questions:     {n}")
    print(f"  Predictions with no number: {n_pred_none} ({n_pred_none/n:.1%})")

    # Most common predictions
    print(f"\n  Top predicted answers:")
    for val, c in count["pred_clean"].value_counts().head(5).items():
        disp = (val[:40] + "...") if len(str(val)) > 40 else val
        print(f"    {disp:45s} {c:6d} ({c/n:.1%})")

    # Confusion matrix over integer values 0-10 (drop non-numeric preds)
    valid = count.dropna(subset=["pred_int", "true_int"]).copy()
    valid = valid[(valid["pred_int"] <= 10) & (valid["true_int"] <= 10)]
    valid["pred_int"] = valid["pred_int"].astype(int)
    valid["true_int"] = valid["true_int"].astype(int)

    labels = list(range(0, 11))
    cm = pd.crosstab(valid["true_int"], valid["pred_int"],
                     rownames=["true"], colnames=["pred"]) \
           .reindex(index=labels, columns=labels, fill_value=0)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("True count")
    ax.set_title("Count confusion matrix\n(BLIP-2 collapses onto one column)")
    # annotate
    mx = cm.values.max()
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm.values[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > mx*0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="count")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "count_confusion_matrix.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {p1}")

    # Distribution comparison: pred vs true
    fig, ax = plt.subplots(figsize=(8, 4))
    true_dist = count["true_int"].value_counts().reindex(labels, fill_value=0)
    pred_dist = count["pred_int"].value_counts().reindex(labels, fill_value=0)
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w/2, true_dist.values, w, label="True answers", color="#4C72B0")
    ax.bar(x + w/2, pred_dist.values, w, label="Model predictions", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Count value")
    ax.set_ylabel("Number of questions")
    ax.set_title("Count: true answer distribution vs. model prediction distribution")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "count_distribution_comparison.png")
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p2}")

    # The key number: how concentrated are predictions?
    top_pred_frac = pred_dist.max() / pred_dist.sum() if pred_dist.sum() else 0
    print(f"\n  → Model's single most common count prediction accounts for "
          f"{top_pred_frac:.1%} of all numeric answers.")
    print(f"    (If the model were truly counting, this would match the true "
          f"distribution, not spike on one value.)")


# Analysis 2: complexity paradox

def complexity_analysis(df, out_dir):
    print(f"\n{'='*55}")
    print("  COMPLEXITY PARADOX ANALYSIS")
    print(f"{'='*55}")

    df = df.copy()
    df["is_correct"] = df["correct"].astype(str).str.lower() == "true"

    # Overall accuracy by complexity (the paradox)
    overall = df.groupby("question_complexity")["is_correct"].mean()
    print("\n  Overall accuracy by complexity (the paradox):")
    for comp, acc in overall.items():
        print(f"    {comp:10s}: {acc:.3f}")

    # Which question types dominate each complexity bucket?
    print("\n  Question-type composition of each complexity bucket:")
    comp_type = pd.crosstab(df["question_complexity"], df["question_type"],
                            normalize="index")
    for comp in ["simple", "medium", "complex"]:
        if comp in comp_type.index:
            top = comp_type.loc[comp].sort_values(ascending=False).head(3)
            parts = ", ".join(f"{t} {v:.0%}" for t, v in top.items())
            print(f"    {comp:10s}: {parts}")

    # The key test: accuracy by complexity WITHIN each question type.
    # If the paradox is a confound, it should vanish (or reverse) here.
    print("\n  Accuracy by complexity, CONTROLLING for question type:")
    print("  (if complex>simple disappears here, the paradox was a confound)")
    pivot = df.pivot_table(index="question_type", columns="question_complexity",
                           values="is_correct", aggfunc="mean")
    pivot = pivot[["simple", "medium", "complex"]] if all(
        c in pivot.columns for c in ["simple", "medium", "complex"]) else pivot
    print(pivot.round(3).to_string())

    # Plot: grouped bars, accuracy by type split by complexity
    types_with_all = pivot.dropna(how="any")
    if len(types_with_all) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(types_with_all))
        cols = [c for c in ["simple", "medium", "complex"] if c in types_with_all.columns]
        w = 0.8 / len(cols)
        colors = {"simple": "#55A868", "medium": "#4C72B0", "complex": "#C44E52"}
        for i, c in enumerate(cols):
            ax.bar(x + (i - len(cols)/2 + 0.5)*w, types_with_all[c].values, w,
                   label=c, color=colors.get(c))
        ax.set_xticks(x)
        ax.set_xticklabels(types_with_all.index, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by question type, split by complexity\n"
                     "(complexity effect within type is small — paradox is a confound)")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        p3 = os.path.join(out_dir, "complexity_paradox.png")
        fig.savefig(p3, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {p3}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")

    count_analysis(df, args.out)
    complexity_analysis(df, args.out)
    print(f"\nDone. Figures in {args.out}/\n")


if __name__ == "__main__":
    main()