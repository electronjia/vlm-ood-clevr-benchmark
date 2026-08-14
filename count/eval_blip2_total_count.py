"""Evaluate BLIP-2 on one generic counting question per CLEVR image.

The counterpart to eval_blip2_counting.py: same model, same prompt format, same
confidence measures, same table and report, so the two runs are directly
comparable. Prompting, count parsing, table writing and reporting are imported
from that module rather than copied, so the metrics cannot drift apart.

What differs is the question. Instead of the several attribute-filtered counting
questions CLEVR poses about each image ("How many cyan metal cylinders are
there?"), every image is asked the same unfiltered "How many objects are there?"
-- once per image, so a row here is an image rather than a question. That asks
only whether the model can enumerate a scene, with no attribute binding in the
way, which is the floor the filtered questions build on.

CLEVR has no annotation for that question, so ground truth is the length of the
scene's object list in CLEVR_<split>_scenes.json (3 to 10 objects per image).
Only train and val have scene annotations; test does not.

One thing cannot be copied from eval_blip2_counting.py. Asked a filtered CLEVR
question, BLIP-2 answers with a bare number, so its confidence is read off
generation step 0; asked this one it answers in prose ("There are 3 objects"),
where step 0 is "There" and carries no count. Confidence is therefore read at
the first step that states a count, reusing `count_step` from
eval_llava_counting.py, which is there for the same reason. That keeps the
`confidence` column meaning the same thing in all three result tables.

`--output` writes a per-image table (see TABLE_COLUMNS) for post-analysis.
"""

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from blip2 import Blip2Runner
from eval_blip2_counting import (
    DEFAULT_CLEVR_ROOT,
    answer_text,
    candidate_token_ids,
    parse_count,
    report,
    write_table,
)
from eval_llava_counting import count_step

QUESTION = "How many objects are there?"


@lru_cache(maxsize=2)
def load_scenes(clevr_root, split="val"):
    """Every scene as (image number, image filename, object count), image order.

    The scene file is the only place the total object count is recorded --
    the question annotations answer filtered counts, not this one. Cached
    because the file is ~34 MB for val and ~160 MB for train.
    """
    scenes_path = Path(clevr_root) / "scenes" / f"CLEVR_{split}_scenes.json"
    with open(scenes_path) as f:
        scenes = json.load(f)["scenes"]

    return tuple(sorted(
        (s["image_index"], s["image_filename"], len(s["objects"])) for s in scenes
    ))


@torch.no_grad()
def predict_batch(runner, image_paths, questions, max_new_tokens=10):
    """Ask one question per image, batched. Same contract as the counting script's.

    Returns (text, confidence, cand_mean, cand_min, cand_std) per row.

    Identical to eval_blip2_counting.predict_batch except for the step the
    confidence is read at: `count_step` finds the first generated token that
    states a count, because a prose answer does not start with one (see the
    module docstring). The candidate tokens still carry a leading space, which
    is how OPT tokenizes a number mid-sentence as well as right after "Answer:".
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]
    prompts = [f"Question: {q} Answer:" for q in questions]

    inputs = runner.processor(
        images=images, text=prompts, padding=True, return_tensors="pt"
    ).to(runner.device, runner.dtype)
    output = runner.model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        output_scores=True, return_dict_in_generate=True,
    )

    tokenizer = runner.processor.tokenizer
    texts = runner.processor.batch_decode(output.sequences, skip_special_tokens=True)
    generated = output.sequences[:, -len(output.scores):]              # (batch, steps)

    probs = torch.stack(output.scores, dim=1).float().softmax(dim=-1)  # (batch, steps, vocab)
    steps = [count_step(tokenizer, row.tolist()) for row in generated]
    rows = torch.arange(len(steps))
    step_probs = probs[rows, torch.tensor(steps)]                      # (batch, vocab)

    emitted = generated[rows, torch.tensor(steps)]                     # (batch,)
    confidence = step_probs.gather(1, emitted[:, None]).squeeze(1)     # (batch,)
    candidates = step_probs[:, candidate_token_ids(tokenizer)]         # (batch, 12)

    return list(zip(
        texts,
        confidence.tolist(),
        candidates.mean(dim=-1).tolist(),
        candidates.min(dim=-1).values.tolist(),
        candidates.std(dim=-1, unbiased=False).tolist(),
    ))


@torch.no_grad()
def evaluate(clevr_root=DEFAULT_CLEVR_ROOT, split="val",
             model_name="Salesforce/blip2-opt-2.7b", batch_size=16, limit=None,
             output_path=None, report_percent=1):
    clevr_root = Path(clevr_root)
    scenes = load_scenes(str(clevr_root), split)
    if limit:
        scenes = scenes[:limit]

    image_dir = clevr_root / "images" / split
    print(f"Asking {QUESTION!r} of {len(scenes)} {split} images with {model_name}\n")

    runner = Blip2Runner(model_name=model_name)
    runner.processor.tokenizer.padding_side = "left"

    records = []
    correct = 0
    start = time.time()

    total = len(scenes)
    step = max(1, total * report_percent // 100)  # log once per report_percent of the data
    next_report = step

    for i in range(0, total, batch_size):
        batch = scenes[i:i + batch_size]
        image_paths = [image_dir / image_name for _, image_name, _ in batch]
        results = predict_batch(runner, image_paths, [QUESTION] * len(batch))

        for (image_number, image_name, expected), (
                text, confidence, cand_mean, cand_min, cand_std) in zip(batch, results):
            processed = parse_count(text)
            is_correct = processed == expected
            correct += is_correct
            records.append({
                "index": image_number,
                "image": image_name,
                "question": QUESTION,
                "ground_truth": expected,
                "guessed_answer_raw": answer_text(text),
                "guessed_answer_processed": processed,
                "confidence": confidence,
                "confidence_mean": cand_mean,
                "confidence_min": cand_min,
                "confidence_std": cand_std,
                "correct": is_correct,
            })

        done = len(records)
        if done >= next_report or done == total:
            rate = done / (time.time() - start)
            remaining = (total - done) / rate if rate else 0
            latest = records[-1]
            print(f"  [{done / total:4.0%}] {done}/{total}  accuracy so far: {correct / done:.2%}"
                  f"  ({rate:.1f} img/s, ~{remaining / 60:.1f} min left)")
            print(f"      latest img: {latest['image']}")
            print(f"      truth     : {latest['ground_truth']}")
            print(f"      model     : {latest['guessed_answer_raw']!r}"
                  f" -> {latest['guessed_answer_processed']}"
                  f"  (conf {latest['confidence']:.3f}, sd {latest['confidence_std']:.3f})",
                  flush=True)
            next_report = (done // step + 1) * step

    elapsed = time.time() - start
    report(records, elapsed)

    if output_path:
        write_table(records, output_path)
        print(f"\nPer-image table written to {output_path}")

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clevr-root", default=str(DEFAULT_CLEVR_ROOT), help="path to CLEVR_v1.0")
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="split with scene annotations")
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    parser.add_argument("--output", default=None,
                        help="write the per-image table here (.csv, or .json for JSON)")
    parser.add_argument("--report-percent", type=int, default=1,
                        help="print a progress line every N%% of the dataset")
    args = parser.parse_args()

    evaluate(
        clevr_root=args.clevr_root,
        split=args.split,
        model_name=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        output_path=args.output,
        report_percent=args.report_percent,
    )


if __name__ == "__main__":
    main()
