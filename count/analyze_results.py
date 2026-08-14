"""Per-value breakdown of a BLIP-2 counting run.

Reads the table written by `eval_blip2_counting.py --output`, then reports
accuracy, mean confidence and mean confidence spread for each answer value --
grouped either by ground truth or by what the model predicted. The two
groupings answer different questions:

    by="truth"      rows are true counts     -> accuracy is RECALL
                    "of the questions whose answer is 3, how many did it get?"
    by="predicted"  rows are guessed counts  -> accuracy is PRECISION
                    "when it said 3, how often was 3 right?"

Same three measures either way, so the two views line up side by side.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from eval_blip2_counting import OUT_OF_DICT

GROUPINGS = {"truth": "ground_truth", "predicted": "guessed_answer_processed"}

ACCURACY_MEANS = {
    "truth": "recall -- share of questions with this true count answered correctly",
    "predicted": "precision -- share of these guesses that were right",
}

# Short axis label for the non-numeric bucket, which is too wide to sit under a tick.
OUT_OF_DICT_LABEL = "no count"

SUMMARY_COLUMNS = ["value", "n", "accuracy", "confidence", "confidence_std"]


# ---------------------------------------------------------------- palette
# One accent hue (categorical slot 1) plus a de-emphasis gray for low-support
# groups -- the "emphasis" form, not a two-colour categorical palette. The
# accent validates clean in both modes (lightness band, chroma floor, >=3:1 on
# its surface); accent-vs-gray separates at OKLab dE 15.9 under protanopia and
# 17.8 unsimulated, well past the 8 / 15 gates.
LIGHT = {
    "surface": "#fcfcfb", "accent": "#2a78d6", "muted_bar": "#c3c2b7",
    "ink": "#0b0b0b", "ink2": "#52514e", "ink3": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
}
DARK = {
    "surface": "#1a1a19", "accent": "#3987e5", "muted_bar": "#52514e",
    "ink": "#ffffff", "ink2": "#c3c2b7", "ink3": "#898781",
    "grid": "#2c2c2a", "axis": "#383835",
}
THEMES = {"light": LIGHT, "dark": DARK}


# ---------------------------------------------------------------- loading
def _as_answer(text):
    """Parse a processed answer cell: an int, or the out-of-dictionary marker."""
    return text if text == OUT_OF_DICT else int(text)


def load_table(path):
    """Read the eval CSV/JSON table, typed, with `correct` recomputed."""
    path = Path(path)
    if path.suffix == ".json":
        import json
        rows = json.loads(path.read_text())
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        processed = _as_answer(str(row["guessed_answer_processed"]))
        truth = int(row["ground_truth"])
        records.append({
            "index": int(row["index"]),
            "question": row["question"],
            "ground_truth": truth,
            "guessed_answer_raw": row["guessed_answer_raw"],
            "guessed_answer_processed": processed,
            "confidence": float(row["confidence"]),
            "confidence_std": float(row["confidence_std"]),
            "correct": processed == truth,
        })
    return records


# ---------------------------------------------------------------- summarise
def _sort_key(value):
    """Numeric values in order, the out-of-dictionary bucket last."""
    return (1, 0) if value == OUT_OF_DICT else (0, value)


def summarize(records, by="truth"):
    """Average accuracy, confidence and confidence spread per answer value.

    `by` is "truth" (group on ground_truth) or "predicted" (group on the
    model's processed answer). Returns rows sorted by value, each:

        {value, n, accuracy, confidence, confidence_std}

    `accuracy` is recall under by="truth" and precision under by="predicted";
    `confidence`/`confidence_std` are means of the per-question columns, so
    they read the same way in both views.
    """
    if by not in GROUPINGS:
        raise ValueError(f"by must be one of {sorted(GROUPINGS)}, got {by!r}")
    key = GROUPINGS[by]

    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)

    return [
        {
            "value": value,
            "n": len(rows),
            "accuracy": fmean(r["correct"] for r in rows),
            "confidence": fmean(r["confidence"] for r in rows),
            "confidence_std": fmean(r["confidence_std"] for r in rows),
        }
        for value, rows in sorted(groups.items(), key=lambda kv: _sort_key(kv[0]))
    ]


# ---------------------------------------------------------------- text view
def print_report(summary, by, min_support=5):
    """The table view: every plotted number, readable without the figure."""
    total = sum(row["n"] for row in summary)
    label = "true count" if by == "truth" else "predicted"

    print(f"\nBy {label}  ({total} questions)")
    print(f"  accuracy = {ACCURACY_MEANS[by]}")
    print(f"  {'value':>17}  {'n':>6}  {'accuracy':>9}  {'conf':>7}  {'conf sd':>8}")
    print("  " + "-" * 54)
    for row in summary:
        thin = " *" if row["n"] < min_support else ""
        print(f"  {str(row['value']):>17}  {row['n']:6d}  {row['accuracy']:8.1%}"
              f"  {row['confidence']:7.4f}  {row['confidence_std']:8.4f}{thin}")
    if any(row["n"] < min_support for row in summary):
        print(f"  * fewer than {min_support} questions -- accuracy here is noise")


# ---------------------------------------------------------------- figure
def plot_summary(summary, by, output_path, theme="light", min_support=5, title=None):
    """Three stacked panels sharing the value axis, one per measure.

    Deliberately not one plot: accuracy runs 0-1 while confidence sits near
    0.09 and its spread near 0.02, and putting those on a shared or twinned
    axis would flatten two of the three. Separate panels, one scale each.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    c = THEMES[theme]
    values = [str(row["value"]) if row["value"] != OUT_OF_DICT else OUT_OF_DICT_LABEL
              for row in summary]
    x = range(len(summary))
    colors = [c["accent"] if row["n"] >= min_support else c["muted_bar"] for row in summary]
    thin_any = any(row["n"] < min_support for row in summary)
    # Keep bars thin: with only a handful of groups a fixed fraction of the axis
    # turns into big saturated blocks.
    width = 0.62 * min(1.0, len(summary) / 8)

    panels = [
        ("Accuracy", [r["accuracy"] for r in summary], True),
        ("Mean confidence in the answer token", [r["confidence"] for r in summary], False),
        ("Mean confidence spread across the 12 candidate answers",
         [r["confidence_std"] for r in summary], False),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.4), sharex=True, dpi=200)
    fig.patch.set_facecolor(c["surface"])

    for ax, (name, heights, is_pct) in zip(axes, panels):
        ax.set_facecolor(c["surface"])
        ax.bar(x, heights, width=width, color=colors, zorder=3)
        ax.set_title(name, color=c["ink2"], fontsize=10, loc="left", pad=8)

        ax.grid(axis="y", color=c["grid"], linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(c["axis"])
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(colors=c["ink3"], labelsize=9, length=0)

        if is_pct:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            # Headline measure, few enough bars to label directly.
            for xi, h in zip(x, heights):
                ax.text(xi, h + 0.03, f"{h:.0%}", ha="center", va="bottom",
                        color=c["ink2"], fontsize=8)
        else:
            ax.set_ylim(0, max(heights) * 1.25 or 1)

    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels([f"{v}\nn={row['n']}" for v, row in zip(values, summary)],
                             color=c["ink3"], fontsize=9)
    axes[-1].set_xlabel("true count" if by == "truth" else "predicted answer",
                        color=c["ink2"], fontsize=10, labelpad=8)

    if thin_any:
        axes[0].legend(
            handles=[Patch(facecolor=c["accent"], label=f"n ≥ {min_support}"),
                     Patch(facecolor=c["muted_bar"], label=f"n < {min_support} (low support)")],
            loc="upper right", frameon=False, fontsize=8, labelcolor=c["ink2"])

    total = sum(row["n"] for row in summary)
    fig.suptitle(title or f"BLIP-2 counting, grouped by "
                          f"{'true count' if by == 'truth' else 'predicted answer'}",
                 color=c["ink"], fontsize=13, x=0.055, ha="left", y=0.978)
    fig.text(0.055, 0.947, f"{total} questions · accuracy = {ACCURACY_MEANS[by]}",
             color=c["ink2"], fontsize=9.5, ha="left")
    fig.text(0.055, 0.022,
             f"Confidence is the softmax probability BLIP-2 put on the token it emitted. "
             f"Spread is the std of that same distribution\nover the 12 candidate answers "
             f"(0-10 and \"None\"); higher means the model separated the options, "
             f"near-zero means it was indifferent.",
             color=c["ink3"], fontsize=8, ha="left")

    fig.subplots_adjust(top=0.875, bottom=0.135, left=0.075, right=0.975, hspace=0.34)
    fig.savefig(output_path, facecolor=c["surface"])
    plt.close(fig)
    return output_path


def write_summary(summary, path):
    """Write the summary table itself, for downstream querying."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary)


# ---------------------------------------------------------------- cli
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", help="CSV/JSON written by eval_blip2_counting.py --output")
    parser.add_argument("--by", default="both", choices=["truth", "predicted", "both"])
    parser.add_argument("--theme", default="light", choices=["light", "dark"])
    parser.add_argument("--min-support", type=int, default=5,
                        help="groups below this many questions are drawn de-emphasised")
    parser.add_argument("--plot-dir", default=None,
                        help="write <plot-dir>/by_<grouping>.png (omit to skip plotting)")
    parser.add_argument("--summary-dir", default=None,
                        help="write <summary-dir>/summary_by_<grouping>.csv")
    args = parser.parse_args()

    records = load_table(args.table)
    groupings = ["truth", "predicted"] if args.by == "both" else [args.by]

    for by in groupings:
        summary = summarize(records, by=by)
        print_report(summary, by, min_support=args.min_support)

        if args.summary_dir:
            path = Path(args.summary_dir) / f"summary_by_{by}.csv"
            write_summary(summary, path)
            print(f"  -> {path}")
        if args.plot_dir:
            path = Path(args.plot_dir) / f"by_{by}.png"
            plot_summary(summary, by, path, theme=args.theme, min_support=args.min_support)
            print(f"  -> {path}")


if __name__ == "__main__":
    main()
