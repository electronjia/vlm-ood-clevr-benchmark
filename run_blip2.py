"""
run_baseline.py — Evaluate pretrained BLIP-2 (OPT-2.7B) on CLEVR (zero-shot).

Reads a CSV of questions, runs BLIP-2 in BATCHES, writes a results CSV.

Input CSV columns:   image_index, image_filename, split, question, answer
Output CSV columns:  image_index, image_filename, split, question, answer, prediction, correct

Batching: instead of running one image per forward pass (which leaves the
GPU mostly idle), we stack several images+questions into one batch and run
model.generate() once per batch. This is typically 5-10x faster.

Usage:
    # Quick test on first 50 rows
    python run_baseline.py --csv val_dataframe.csv --limit 100

    # Full run with a bigger batch (tune to your GPU)
    python run_baseline.py --csv val_dataframe.csv --batch-size 16

    # Custom output path
    python run_baseline.py --csv val_dataframe.csv --out my_results.csv
"""

import argparse
import csv
import os

import torch
from PIL import Image

from metrics import soft_match, exact_match
from config import *

BLIP2_ID = "Salesforce/blip2-opt-2.7b"


def load_blip2():
    """
    Load BLIP-2 with the OPT-2.7B language model.

    At 2.7B parameters this fits in fp16 on most GPUs (~6GB).
    device_map="auto" places it on the GPU if one is available.
    use_fast=True uses the faster image processor.
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    print(f"Loading BLIP-2 ({BLIP2_ID})...")

    processor = Blip2Processor.from_pretrained(BLIP2_ID, use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_ID,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # For batching we need a pad token and left-padding (so generated
    # tokens line up on the right for every sequence in the batch).
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    print("  Loaded!\n")
    return model, processor


@torch.inference_mode()
def ask_blip2_batch(model, processor, images, questions):
    """
    Ask BLIP-2 a BATCH of questions, one per image. Zero-shot.

    Args:
        images:    list of PIL Images (length B)
        questions: list of question strings (length B, aligned with images)

    Returns:
        list of B predicted answer strings, in the same order.

    BLIP-2 has no chat template — each prompt is "Question: ... Answer:".
    The processor handles a list of images + list of prompts and pads them
    into one batch tensor.
    """
    device = next(model.parameters()).device

    prompts = [f"Question: {q} Answer:" for q in questions]

    # Processor takes parallel lists: images and text, padded into a batch.
    inputs = processor(
        images=images, text=prompts,
        return_tensors="pt", padding=True,
    ).to(device, torch.float16)

    output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    # batch_decode returns one string per row in the batch.
    answers = processor.batch_decode(output_ids, skip_special_tokens=True)
    return [a.strip() for a in answers]


def load_csv(csv_path, limit=None):
    """Read input CSV into a list of dicts."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # if row["question_type"] == cfg_question_type:
            rows.append({
                "image_filename": row["image_filename"].strip(),
                "image_index": row["image_index"],
                "split": row["split"],
                "question_type": row["question_type"],
                "question_complexity": row["question_complexity"],
                "question": row["question"].strip(),
                "answer": row["answer"].strip(),
            })
            if limit and len(rows) >= limit:
                break
    return rows


def chunked(seq, size):
    """Yield successive chunks of `size` from `seq`."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLIP-2 OPT-2.7B on CLEVR (zero-shot, batched)")
    parser.add_argument("--csv", required=True, help="Input CSV (image_index, image_filename, split, question, answer)")
    parser.add_argument("--images", default=CLEVR_IMAGES, help="Folder containing CLEVR images")
    parser.add_argument("--out", default=None, help="Output CSV path (default: results/blip2_clevr_results.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N rows (quick test)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Images per forward pass. Bigger = faster but more VRAM. "
                             "Try 8, then raise until nvidia-smi shows the GPU near full.")
    args = parser.parse_args()

    # -- Load questions --
    print(f"Reading questions from {args.csv} ...")
    rows = load_csv(args.csv, limit=args.limit)
    print(f"  Loaded {len(rows)} questions"
          + (f" (limited to first {args.limit})" if args.limit else "")
          + f"  |  batch size = {args.batch_size}\n")

    # -- Load model --
    model, processor = load_blip2()

    # -- Output setup --
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.out or os.path.join(RESULTS_DIR, "blip2_clevr_results.csv")
    fieldnames = ["image_index", "image_filename", "split", "question_type", "question_complexity", "correct", "question", "answer", "prediction"]


    soft_correct = 0
    exact_correct = 0
    done = 0

    # We write batch-by-batch so a crash midway doesn't lose completed work.
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for batch in chunked(rows, args.batch_size):
            # Load all images for this batch
            images = [
                Image.open(os.path.join(args.images, r["image_filename"])).convert("RGB")
                for r in batch
            ]
            questions = [r["question"] for r in batch]

            # One forward pass for the whole batch
            predictions = ask_blip2_batch(model, processor, images, questions)

            # Score and write each row in the batch
            for r, prediction in zip(batch, predictions):
                is_correct = soft_match(prediction, r["answer"])
                soft_correct += int(is_correct)
                exact_correct += int(exact_match(prediction, r["answer"]))

                writer.writerow({
                    "image_index": r["image_index"],
                    "image_filename": r["image_filename"],
                    "split": r["split"],
                    "question_type": r["question_type"],
                    "question_complexity": r["question_complexity"],
                    "correct": is_correct,     # True / False
                    "question": r["question"],
                    "answer": r["answer"],
                    "prediction": prediction,
                    
                })

            f_out.flush()   # persist after each batch
            done += len(batch)
            print(f"  {done}/{len(rows)}  "
                  f"(last batch e.g. pred='{predictions[-1]}' gt='{batch[-1]['answer']}')")

    n = len(rows)
    print(f"\n{'='*50}")
    print("  BLIP-2 OPT-2.7B ON CLEVR — zero-shot (batched)")
    print(f"{'='*50}")
    print(f"  Questions:       {n}")
    print(f"  Soft accuracy:   {soft_correct}/{n} = {soft_correct/n:.3f}")
    print(f"  Exact accuracy:  {exact_correct}/{n} = {exact_correct/n:.3f}")
    print(f"{'='*50}")
    print(f"\n  Results written to {out_path}\n")


if __name__ == "__main__":
    main()