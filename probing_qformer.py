"""
probe_count.py — Linear probing: does the Q-Former bottleneck lose counting info?

THE EXPERIMENT
--------------
BLIP-2 compresses the image: vision encoder (257 tokens) -> Q-Former (32 tokens). 
Count question type shows the worst performance when using BLIP-2. 
Since BLIP-2 is question-agnostic and only epends on the image. Count question type is the best choice in this case.
The reason being is because the question type count only relates to property of the image and answer is a digit.
We ask: is the object-count information preserved through that compression?

Method (linear probing):
  1. For each count-question image, capture BOTH intermediate tensors:
       - vision encoder output  (257 x 1408)
       - Q-Former output         (32 x 768)
  2. Pool each token-set into one fixed vector via MEAN + MAX concat:
       - vision  -> 2*1408 = 2816-dim
       - qformer -> 2*768  = 1536-dim
     (mean captures the average activation; max captures whether ANY single
      token strongly encodes a feature — important for counting, where one
      token may spike per object.)
  3. Train a simple logistic-regression probe on each representation to
     predict the object count.
  4. Compare test accuracy. If vision >> qformer, the compression discarded
     counting information.

Two stages so a long extraction isn't wasted:
    python probing_count_questions.py extract --csv val_dataframe.csv --limit 500
    python probing_count_questions.py train

Usage:
    # Stage 1 — extract pooled vectors + labels (the slow, GPU part)
    python probing_count_questions.py extract --csv val_dataframe.csv            # all count images
    python probing_count_questions.py extract --csv val_dataframe.csv --limit 500  # quick test

    # Stage 2 — train the two probes and compare (fast, CPU)
    python probing_count_questions.py train
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

# Where the extracted arrays get saved (stage 1 -> stage 2 hand-off)
VISION_PATH = os.path.join(RESULTS_DIR,"qformer_probe", "probe_vision_X.npy")
QFORMER_PATH = os.path.join(RESULTS_DIR,"qformer_probe", "probe_qformer_X.npy")
LABELS_PATH = os.path.join(RESULTS_DIR, "qformer_probe","probe_labels_y.npy")


# Pooling: token-set -> one fixed vector

def mean_max_pool(tokens):
    """
    Collapse a (num_tokens, dim) array into a single (2*dim,) vector by
    concatenating the per-feature MEAN and MAX across tokens.

    mean -> overall activation level for each feature
    max  -> whether ANY single token strongly encodes that feature
            (guards against mean-pooling washing out per-object signals)
    """
    tokens = np.squeeze(np.asarray(tokens, dtype=np.float32))
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D (tokens, dim), got {tokens.shape}")
    mean_vec = tokens.mean(axis=0)   # (dim,)
    max_vec = tokens.max(axis=0)     # (dim,)
    return np.concatenate([mean_vec, max_vec])   # (2*dim,)


# Stage 1 — extract pooled vectors + count labels

def load_blip2():
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"Loading BLIP-2 ({BLIP2_ID})...")
    processor = Blip2Processor.from_pretrained(BLIP2_ID, use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_ID, dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda")
    model.eval()
    # Sanity: make sure nothing landed on the meta device
    devices = {p.device.type for p in model.parameters()}
    print(f"  model devices: {devices}")
    if "meta" in devices:
        raise RuntimeError("Model has meta tensors — weights not fully loaded!")
    print("  Loaded!\n")
    return model, processor


def extract(csv_path, limit=None):
    """
    Run BLIP-2 over unique count-question images, pool both tensors,
    save X_vision, X_qformer, y to disk.
    """
    import torch
    from PIL import Image
    from tqdm import tqdm

    df = pd.read_csv(csv_path)

    # Keep only count questions
    df = df[df["question_type"] == "count"].copy()
    print(f"count questions: {len(df)}")

    # Q-Former is question-agnostic, so ONE forward pass per unique image.
    df = df.drop_duplicates(subset="image_filename").reset_index(drop=True)
    print(f"unique count images: {len(df)}")

    # LABEL = total objects in the image (image-level property, from num_objects).
    # This is what makes the probe valid: the Q-Former is question-agnostic,
    # so the label must depend on the image alone, not on the question.
    df["count_label"] = df["num_objects"].astype(int)
    print(f"images with total-count label: {len(df)}")
    print(f"  total-count range: {df['count_label'].min()}–{df['count_label'].max()}")

    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
        print(f"(limited to first {limit})")

    model, processor = load_blip2()
    device = next(model.parameters()).device

    # Register hooks ONCE (they overwrite `captured` each pass)
    captured = {}

    def make_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                captured[name] = output[0]
            elif hasattr(output, "last_hidden_state"):
                captured[name] = output.last_hidden_state
            else:
                captured[name] = output
        return hook

    h1 = model.vision_model.register_forward_hook(make_hook("vision"))
    h2 = model.qformer.register_forward_hook(make_hook("qformer"))

    X_vision, X_qformer, y = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(CLEVR_IMAGES, row["image_filename"])
        image = Image.open(img_path).convert("RGB")

        # A simple prompt — the Q-Former ignores it anyway, and only
        # need the forward pass to reach the vision encoder + Q-Former.
        prompt = "Question: How many objects are in the image? Answer:"
        inputs = processor(image, prompt, return_tensors="pt").to(device, torch.float16)

        with torch.inference_mode():
            # max_new_tokens=1 —  only need the forward pass up to the
            # Q-Former; e don't care about the generated text.
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)

        # Pool both captured tensors (move to CPU float32 first)
        vision_tok = captured["vision"][0].float().cpu().numpy()   # (257, 1408)
        qformer_tok = captured["qformer"][0].float().cpu().numpy() # (32, 768)

        X_vision.append(mean_max_pool(vision_tok))
        X_qformer.append(mean_max_pool(qformer_tok))
        y.append(int(row["count_label"]))

    h1.remove()
    h2.remove()

    X_vision = np.stack(X_vision)
    X_qformer = np.stack(X_qformer)
    y = np.array(y)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # print(f"DID NOT SAVE!!")
    np.save(VISION_PATH, X_vision)
    np.save(QFORMER_PATH, X_qformer)
    np.save(LABELS_PATH, y)

    print(f"\nSaved:")
    print(f"  vision  X: {X_vision.shape}  -> {VISION_PATH}")
    print(f"  qformer X: {X_qformer.shape}  -> {QFORMER_PATH}")
    print(f"  labels  y: {y.shape}  (counts {y.min()}-{y.max()})  -> {LABELS_PATH}")


# Stage 2 — train the two probes and compare

def train_probe(X, y, name):
    """
    Train a logistic-regression probe and report test accuracy.

    Returns (exact_accuracy, within1_accuracy).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features, then logistic regression.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    exact = (pred == y_te).mean()
    within1 = (np.abs(pred - y_te) <= 1).mean()

    print(f"\n── {name} probe ──")
    print(f"  feature dim:       {X.shape[1]}")
    print(f"  exact accuracy:    {exact:.3f}")
    print(f"  within ±1 accuracy: {within1:.3f}")
    return exact, within1


def train():
    """Load extracted vectors, train both probes, compare."""
    for p in [VISION_PATH, QFORMER_PATH, LABELS_PATH]:
        if not os.path.exists(p):
            print(f"Missing {p}. Run the 'extract' stage first.")
            return

    X_vision = np.load(VISION_PATH)
    X_qformer = np.load(QFORMER_PATH)
    y = np.load(LABELS_PATH)

    y = np.minimum(y, 7)   # ← ADD: merge 7,8,9,10 into a single "7+" class

    
    print(f"Loaded {len(y)} images. Count distribution:")
    vals, counts = np.unique(y, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  count={v}: {c}")

    # Majority-class baseline — the floor any probe must beat.
    majority = counts.max() / counts.sum()
    print(f"\nMajority-class baseline: {majority:.3f}")

    v_exact, v_w1 = train_probe(X_vision, y, "VISION encoder (257 tokens)")
    q_exact, q_w1 = train_probe(X_qformer, y, "Q-FORMER (32 tokens)")

    print(f"\n{'='*55}")
    print("  COMPARISON — counting information through the bottleneck")
    print(f"{'='*55}")
    print(f"  Vision  exact: {v_exact:.3f}   within±1: {v_w1:.3f}")
    print(f"  Q-Former exact: {q_exact:.3f}   within±1: {q_w1:.3f}")
    print(f"  Drop (exact):   {v_exact - q_exact:+.3f}")
    print(f"{'='*55}")
    if q_exact < v_exact - 0.05:
        print("  → Q-Former probe is notably worse: the 32-token compression")
        print("    appears to DISCARD counting information.")
    else:
        print("  → Q-Former probe roughly matches vision: counting information")
        print("    largely SURVIVES the compression.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Linear probing: count info through Q-Former")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_ext = sub.add_parser("extract", help="Stage 1: extract pooled tensors")
    p_ext.add_argument("--csv", required=True, default="val_dataframe.csv", help="CSV with image_filename, question_type, answer")
    p_ext.add_argument("--limit", type=int, default=None, help="Only first N images (quick test)")

    sub.add_parser("train", help="Stage 2: train probes and compare")

    args = parser.parse_args()

    if args.stage == "extract":
        extract(args.csv, limit=args.limit)
    elif args.stage == "train":
        train()


if __name__ == "__main__":
    main()