"""
probing_llm.py — Trace count-decodability through BLIP-2's language model.

FOLLOW-UP TO probe_count.py
---------------------------
The Q-Former probe showed TOTAL OBJECT COUNT is recoverable (~0.81) from the
32 compressed query tokens. So the bottleneck did NOT destroy count info.
This script asks: does that information survive DEEPER — through the
language_projection and into OPT's decoder layers — or does OPT discard it?

We probe TOTAL OBJECT COUNT (num_objects) — the SAME image-level label as the
Q-Former probe — at several depths, using the SAME mean+max pooling. Only the
extraction depth changes, so the accuracy-vs-depth curve is a controlled
measurement of where (if anywhere) count stops being linearly decodable.

Depth points captured in ONE forward pass per image:
  1. qformer            — Q-Former output              (32 x 768)   [baseline]
  2. projection         — after language_projection     (32 x 2560)
  3. opt_L0             — OPT decoder layer 0 output, query positions
  4. opt_L8             — OPT decoder layer 8
  5. opt_L16            — OPT decoder layer 16
  6. opt_last           — OPT final decoder layer

CRITICAL DETAIL: BLIP-2 prepends the 32 projected query tokens to the text
tokens before OPT. So in every OPT layer's hidden state (batch, seq, 2560),
the FIRST 32 positions are the visual query tokens. We slice [:, :32, :] to
probe only those. The script prints the sequence layout on the first image
so you can VERIFY the 32 query positions before the full run.

Two stages:
    python probing_llm.py extract --csv val_dataframe.csv --limit 50   # test
    python probing_llm.py extract --csv val_dataframe.csv --n 2500     # real
    python probing_llm.py train
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import numpy as np
import pandas as pd

from config import CLEVR_IMAGES, RESULTS_DIR

BLIP2_ID = "Salesforce/blip2-opt-2.7b"
N_QUERY_TOKENS = 32   # BLIP-2 uses 32 Q-Former query tokens

PROBE_DIR = os.path.join(RESULTS_DIR, "llm_probe")

# Which OPT decoder layers to probe (plus qformer + projection, added separately)
OPT_LAYERS = [0, 8, 16, -1]   # -1 = last

# Depth-point names in order (for consistent plotting)
DEPTH_ORDER = ["qformer", "projection", "opt_L0", "opt_L8", "opt_L16", "opt_last"]


# Pooling (identical to the Q-Former probe)

def mean_max_pool(tokens):
    """(num_tokens, dim) -> (2*dim,) via concatenated mean and max over tokens."""
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D (tokens, dim), got {tokens.shape}")
    return np.concatenate([tokens.mean(axis=0), tokens.max(axis=0)])


# Balanced sampling by total-count

def balanced_sample(df, n, seed=42):
    """
    Sample up to n rows aiming for balance across total-count classes.
    Caps rare high counts into a 8+ bucket for labeling only after.
    """
    rng = np.random.RandomState(seed)
    # unique images
    df = df.drop_duplicates(subset="image_filename").reset_index(drop=True)
    df["count_label"] = df["num_objects"].astype(int)

    classes = sorted(df["count_label"].unique())
    per_class = max(1, n // len(classes))

    parts = []
    for c in classes:
        sub = df[df["count_label"] == c]
        take = min(len(sub), per_class)
        parts.append(sub.sample(take, random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


# Model loading

def load_blip2():
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"Loading BLIP-2 ({BLIP2_ID})...")
    processor = Blip2Processor.from_pretrained(BLIP2_ID, use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_ID, dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda")
    model.eval()
    devices = {p.device.type for p in model.parameters()}
    print(f"  model devices: {devices}")
    if "meta" in devices:
        raise RuntimeError("Model has meta tensors — not fully loaded!")
    print("  Loaded!\n")
    return model, processor


# Stage 1 — extract pooled vectors at every depth

def extract(csv_path, limit=None, n=None, seed=42):
    import torch
    from PIL import Image
    from tqdm import tqdm

    df = pd.read_csv(csv_path)
    df = df[df["question_type"] == "count"].copy()
    print(f"count-question rows: {len(df)}")

    if n:
        df = balanced_sample(df, n, seed)
        print(f"balanced sample: {len(df)} unique images")
    else:
        df = df.drop_duplicates(subset="image_filename").reset_index(drop=True)
        df["count_label"] = df["num_objects"].astype(int)
        print(f"unique images: {len(df)}")

    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
        print(f"(limited to {limit})")

    print("total-count distribution:")
    print(df["count_label"].value_counts().sort_index())

    model, processor = load_blip2()
    device = next(model.parameters()).device

    # ── Register hooks at every depth point ──
    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            captured[name] = out[0] if isinstance(out, tuple) else \
                (out.last_hidden_state if hasattr(out, "last_hidden_state") else out)
        return hook

    handles = []
    handles.append(model.qformer.register_forward_hook(make_hook("qformer")))
    handles.append(model.language_projection.register_forward_hook(make_hook("projection")))

    # OPT decoder layers
    opt_layers = model.language_model.model.decoder.layers
    n_opt = len(opt_layers)
    layer_map = {}
    for L in OPT_LAYERS:
        idx = L if L >= 0 else n_opt - 1
        name = "opt_last" if L == -1 else f"opt_L{L}"
        layer_map[name] = idx
        handles.append(opt_layers[idx].register_forward_hook(make_hook(name)))

    print(f"OPT has {n_opt} decoder layers. Probing: {layer_map}\n")

    # storage
    feats = {name: [] for name in DEPTH_ORDER}
    y = []

    first = True
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image = Image.open(os.path.join(CLEVR_IMAGES, row["image_filename"])).convert("RGB")
        prompt = "Question: How many objects are in the image? Answer:"
        inputs = processor(image, prompt, return_tensors="pt").to(device, torch.float16)

        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)

        # ── First-iteration VERIFICATION of the query-token layout ──
        if first:
            print("\n=== FIRST-ITERATION SHAPE CHECK ===")
            for name in DEPTH_ORDER:
                t = captured[name]
                print(f"  {name:12s}: {tuple(t.shape)}")
            opt0 = captured["opt_L0"]
            print(f"\n  OPT hidden state seq length: {opt0.shape[1]}")
            print(f"  First {N_QUERY_TOKENS} positions = visual query tokens (what we probe)")
            print(f"  Remaining {opt0.shape[1]-N_QUERY_TOKENS} = text prompt tokens (ignored)")
            print("===================================\n")
            first = False

        # ── Pool each depth point ──
        # qformer & projection: already (1, 32, dim) — all 32 are query tokens
        qf = captured["qformer"][0].float().cpu().numpy()          # (32, 768)
        pj = captured["projection"][0].float().cpu().numpy()        # (32, 2560)
        feats["qformer"].append(mean_max_pool(qf))
        feats["projection"].append(mean_max_pool(pj))

        # OPT layers: (1, seq, 2560) — slice FIRST 32 positions (query tokens)
        for name in ["opt_L0", "opt_L8", "opt_L16", "opt_last"]:
            h = captured[name][0, :N_QUERY_TOKENS, :].float().cpu().numpy()  # (32, 2560)
            feats[name].append(mean_max_pool(h))

        y.append(int(row["count_label"]))

    for h in handles:
        h.remove()

    os.makedirs(PROBE_DIR, exist_ok=True)
    for name in DEPTH_ORDER:
        arr = np.stack(feats[name])
        np.save(os.path.join(PROBE_DIR, f"X_{name}.npy"), arr)
        print(f"  saved X_{name}: {arr.shape}")
    np.save(os.path.join(PROBE_DIR, "y.npy"), np.array(y))
    print(f"  saved y: {len(y)} labels")


# Stage 2 — train a probe at each depth, plot the curve

def train_probe(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    exact = (pred == y_te).mean()
    within1 = (np.abs(pred - y_te) <= 1).mean()
    return exact, within1


def train():
    import matplotlib.pyplot as plt

    y = np.load(os.path.join(PROBE_DIR, "y.npy"))
    y = np.minimum(y, 8)   # cap rare high counts

    vals, counts = np.unique(y, return_counts=True)
    majority = counts.max() / counts.sum()
    print(f"Labels: {len(y)}   classes: {dict(zip(vals.tolist(), counts.tolist()))}")
    print(f"Majority baseline: {majority:.3f}\n")

    results = {}
    for name in DEPTH_ORDER:
        path = os.path.join(PROBE_DIR, f"X_{name}.npy")
        if not os.path.exists(path):
            print(f"  (missing {name}, skipping)")
            continue
        X = np.load(path)
        exact, within1 = train_probe(X, y)
        results[name] = (exact, within1)
        print(f"  {name:12s}  exact={exact:.3f}  within±1={within1:.3f}  (dim {X.shape[1]})")

    # Plot the depth curve
    names = [n for n in DEPTH_ORDER if n in results]
    exacts = [results[n][0] for n in names]
    within1s = [results[n][1] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.plot(x, exacts, "o-", label="exact accuracy", color="#4C72B0", lw=2, markersize=7)
    ax.plot(x, within1s, "s-", label="within ±1", color="#55A868", lw=2, markersize=7)
    ax.axhline(majority, ls="--", color="gray", label=f"majority baseline ({majority:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("count-probe accuracy")
    ax.set_xlabel("depth through the language model →")
    ax.set_title("Where does total-count information survive in BLIP-2?\n"
                 "(flat & high = info present throughout, LLM fails to USE it)")
    ax.legend()
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(PROBE_DIR, "count_depth_curve.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved depth curve → {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)
    pe = sub.add_parser("extract")
    pe.add_argument("--csv", required=True)
    pe.add_argument("--limit", type=int, default=None)
    pe.add_argument("--n", type=int, default=None, help="balanced sample size")
    sub.add_parser("train")
    args = p.parse_args()

    if args.stage == "extract":
        extract(args.csv, limit=args.limit, n=args.n)
    else:
        train()


if __name__ == "__main__":
    main()