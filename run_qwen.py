"""
run_qwen.py — Evaluate Qwen2.5-VL on CLEVR for comparison with BLIP-2.

Speed features:
  - Batched inference (many images per forward pass)
  - Auto-detect 4-bit quantization, fall back to fp16
  - Short max_new_tokens (CLEVR answers are 1-3 words)
  - Stratified sampling (balanced across question types) if you don't want all
  - Crash-safe: checkpoints every N rows and RESUMES on restart

Input CSV columns:   image_index, image_filename, split, question_type, question, answer
Output CSV columns:  + prediction, correct

Usage:
    # Stratified ~15k sample, batched
    python run_qwen.py --csv val_dataframe.csv --sample 15000 --batch-size 8

    # Full dataset
    python run_qwen.py --csv val_dataframe.csv --batch-size 8

    # Resume an interrupted run (same command — it skips finished rows)
    python run_qwen.py --csv val_dataframe.csv --sample 15000 --batch-size 8
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from metrics import soft_match, exact_match
from config import CLEVR_IMAGES, RESULTS_DIR, MAX_NEW_TOKENS

QWEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CHECKPOINT_EVERY = 500


# Stratified sampling

def stratified_sample(df, n, seed=42):
    """
    Sample ~n rows, proportionally balanced across question_type.
    If the dataset already has <= n rows, return it unchanged.
    """
    if n is None or len(df) <= n:
        print(f"  dataset has {len(df)} rows (<= requested {n}); using all")
        return df.reset_index(drop=True)

    # proportional allocation per question type
    frac = n / len(df)
    parts = []
    for qtype, group in df.groupby("question_type"):
        take = max(1, int(round(len(group) * frac)))
        take = min(take, len(group))
        parts.append(group.sample(take, random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  stratified sample: {len(out)} rows across "
          f"{df['question_type'].nunique()} question types")
    return out


# Model loading — auto-detect 4-bit, fall back to fp16

def load_qwen():
    from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Loading Qwen ({QWEN_ID})...")

    quant_config = None
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("  bitsandbytes found — using 4-bit quantization")
    except Exception:
        print("  bitsandbytes unavailable — using fp16")

    processor = Qwen2_5_VLProcessor.from_pretrained(
        QWEN_ID, use_fast=True,
        min_pixels=128*28*28,      # floor
        max_pixels=256*28*28,      # cap — fewer visual tokens, much faster
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID,
        dtype=torch.float16,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    # left padding for batched generation
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    print("  Loaded!\n")
    return model, processor


# Batched inference

@torch.inference_mode()
def ask_qwen_batch(model, processor, images, questions):
    """Run a batch of (image, question) pairs. Returns list of answer strings."""
    device = next(model.parameters()).device

    texts = []
    for q in questions:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{q}\nAnswer in one or two words."},
            ],
        }]
        texts.append(processor.apply_chat_template(messages, add_generation_prompt=True))

    inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True
    ).to(device)

    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    # strip the prompt tokens from each row
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    answers = processor.batch_decode(trimmed, skip_special_tokens=True)
    return [a.strip() for a in answers]


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# Main — with resume

def main():
    ap = argparse.ArgumentParser(description="Qwen2.5-VL on CLEVR (batched, resumable)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", default=CLEVR_IMAGES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="Stratified sample size (omit for full dataset)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = args.out or os.path.join(RESULTS_DIR, "qwen_clevr_results.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load and (optionally) sample ──
    print(f"Reading {args.csv} ...")
    df = pd.read_csv(args.csv)
    df = stratified_sample(df, args.sample, seed=args.seed)

    # Give every row a stable id so resume can match completed rows.
    # Use image_filename + question as the key (unique per VQA item).
    df["row_key"] = df["image_filename"].astype(str) + "||" + df["question"].astype(str)

    # ── Resume: load already-completed keys ──
    done_keys = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        if "row_key" in prev.columns:
            done_keys = set(prev["row_key"].tolist())
        print(f"  resuming: {len(done_keys)} rows already done, will skip them")

    todo = df[~df["row_key"].isin(done_keys)].reset_index(drop=True)
    print(f"  {len(todo)} rows to process (of {len(df)} total)\n")

    if len(todo) == 0:
        print("Nothing to do — all rows already completed.")
        summarize(out_path)
        return

    model, processor = load_qwen()

    # ── Write header if file is new ──
    fieldnames = list(df.columns) + ["prediction", "correct"]
    write_header = not os.path.exists(out_path)
    if write_header:
        pd.DataFrame(columns=fieldnames).to_csv(out_path, index=False)

    buffer = []
    processed = 0

    def flush():
        nonlocal buffer
        if buffer:
            pd.DataFrame(buffer).to_csv(out_path, mode="a", header=False, index=False)
            buffer = []

    
    batches = [todo.iloc[i:i+args.batch_size] for i in range(0, len(todo), args.batch_size)]
    pbar = tqdm(batches, desc="Qwen eval", unit="batch")
    running_correct = 0
    running_total = 0

    for batch_df in pbar:
        rows = batch_df.to_dict("records")
        images = [Image.open(os.path.join(args.images, r["image_filename"])).convert("RGB")
                  for r in rows]
        questions = [r["question"] for r in rows]

        preds = ask_qwen_batch(model, processor, images, questions)

        for r, pred in zip(rows, preds):
            r["prediction"] = pred
            is_correct = soft_match(pred, str(r["answer"]))
            r["correct"] = is_correct
            running_correct += int(is_correct)
            running_total += 1
            buffer.append(r)

        processed += len(rows)
        # live accuracy in the bar
        pbar.set_postfix(acc=f"{running_correct/running_total:.3f}")

        if processed % CHECKPOINT_EVERY < args.batch_size:
            flush()
            print(f"  {processed}/{len(todo)} done (checkpointed)")

    flush()
    print(f"\nFinished. Results in {out_path}")
    summarize(out_path)


def summarize(out_path):
    df = pd.read_csv(out_path)
    df["correct"] = df["correct"].astype(str).str.lower() == "true"
    print(f"\n{'='*55}")
    print("  QWEN2.5-VL ON CLEVR")
    print(f"{'='*55}")
    print(f"  Total: {len(df)}   Overall accuracy: {df['correct'].mean():.3f}")
    print(f"\n  By question type:")
    for qt, g in df.groupby("question_type"):
        print(f"    {qt:16s}: {g['correct'].mean():.3f}  (n={len(g)})")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()