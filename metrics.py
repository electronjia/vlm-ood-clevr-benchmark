"""
metrics.py — Compute evaluation metrics.

Two accuracy modes:
  - Exact match:  prediction must match ground truth exactly (case-insensitive)
  - Soft match:   allows synonyms ("yeah" = "yes") and substring
                  ("The color is red" matches ground truth "red")
"""

import re
from collections import defaultdict


# Words that mean the same thing in CLEVR answers
SYNONYMS = {
    "yes": {"yes", "yeah", "yep", "true", "correct"},
    "no":  {"no", "nope", "nah", "false", "incorrect"},
    "0": {"zero", "0", "none"},
    "1": {"one", "1"},  
    "2": {"two", "2"},  
    "3": {"three", "3"},
    "4": {"four", "4"}, 
    "5": {"five", "5"},
    "6": {"six", "6"},
    "7": {"seven", "7"},
    "8": {"eight", "8"},
    "9": {"nine", "9"},
    "10": {"ten", "10"},
    "small": {"small", "tiny", "little"},
    "large": {"large", "big"},
    "metal": {"metal", "metallic", "shiny"},
    "rubber": {"rubber", "matte"},
}

# Build a reverse lookup: "yeah" → "yes", "tiny" → "small", etc.
_CANON = {}
for canonical, synonyms in SYNONYMS.items():
    for word in synonyms:
        _CANON[word] = canonical
    _CANON[canonical] = canonical


def normalize(text):
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def soft_match(prediction, ground_truth):
    """
    Does the prediction match the ground truth semantically?

    Handles three cases:
      1. Exact match after normalization
      2. Both resolve to the same synonym ("yeah" == "yes")
      3. Ground truth appears inside prediction ("The color is red" matches "red")
    """
    pred = normalize(prediction)
    gt = normalize(ground_truth)

    if pred == gt:
        return True
    if _CANON.get(pred) == _CANON.get(gt) and _CANON.get(gt) is not None:
        return True
    if gt in pred:
        return True
    return False


def exact_match(prediction, ground_truth):
    """Strict match after normalization."""
    return normalize(prediction) == normalize(ground_truth)


def compute_metrics(results):
    """
    Compute all metrics from a list of prediction dicts.

    Each dict has: {"question", "answer", "prediction", "type"}

    Returns a dict with:
      - soft_accuracy:   overall soft-match accuracy
      - exact_accuracy:  overall exact-match accuracy
      - per_type:        accuracy broken down by question type
      - n_samples:       total number of predictions
    """
    if not results:
        return {"soft_accuracy": 0, "exact_accuracy": 0, "per_type": {}, "n_samples": 0}

    # Overall accuracy
    soft_correct = sum(soft_match(r["prediction"], r["answer"]) for r in results)
    exact_correct = sum(exact_match(r["prediction"], r["answer"]) for r in results)
    n = len(results)

    # Per question type
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)

    per_type = {}
    for qtype, items in sorted(by_type.items()):
        correct = sum(soft_match(r["prediction"], r["answer"]) for r in items)
        per_type[qtype] = {
            "accuracy": correct / len(items),
            "n": len(items),
        }

    return {
        "soft_accuracy":  soft_correct / n,
        "exact_accuracy": exact_correct / n,
        "per_type":       per_type,
        "n_samples":      n,
    }


def compute_ood_drop(id_accuracy, ood_accuracy):
    """
    How much worse is OOD compared to ID?

    Returns relative drop as a fraction.
    Example: ID=0.8, OOD=0.4  →  drop = 0.5 (50% degradation)
    """
    if id_accuracy == 0:
        return 0.0
    return (id_accuracy - ood_accuracy) / id_accuracy


def print_report(all_results, model_name):
    """
    Print a readable summary of results across splits.

    all_results: dict of split_name → result list
    """
    print(f"\n{'='*55}")
    print(f"  Results for: {model_name}")
    print(f"{'='*55}")

    metrics_by_split = {}
    for split_name, results in all_results.items():
        m = compute_metrics(results)
        metrics_by_split[split_name] = m
        print(f"\n  {split_name} ({m['n_samples']} samples)")
        print(f"    Soft accuracy:  {m['soft_accuracy']:.3f}")
        print(f"    Exact accuracy: {m['exact_accuracy']:.3f}")
        print(f"    Per type:")
        for qtype, info in m["per_type"].items():
            print(f"      {qtype:12s}: {info['accuracy']:.3f}  (n={info['n']})")

    # OOD drops
    id_acc = metrics_by_split.get("id_val", {}).get("soft_accuracy", 0)
    if id_acc > 0:
        print(f"\n  OOD drops (relative to id_val = {id_acc:.3f}):")
        for split_name, m in metrics_by_split.items():
            if split_name.startswith("ood_"):
                drop = compute_ood_drop(id_acc, m["soft_accuracy"])
                print(f"    {split_name:12s}: {drop:+.1%} drop")

    print()
    return metrics_by_split
