"""Evaluate BLIP-2 counting accuracy over the CLEVR counting questions.

Both the question and its answer come from CLEVR_<split>_questions.json, so each
image is asked the questions CLEVR actually poses about it rather than one
generic "how many objects" prompt.

`--output` writes a per-question table (see TABLE_COLUMNS) for post-analysis.
"""

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from blip2 import Blip2Runner

DEFAULT_CLEVR_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "CLEVR_v1.0"

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

# Processed answer for a response that states no count at all -- not a zero.
OUT_OF_DICT = "out of dictionary"

# The answer space CLEVR counting questions can take, plus the "no answer" case.
# BLIP-2 generates after "Answer:", so the candidate strings carry a leading space.
CANDIDATE_ANSWERS = [str(n) for n in range(11)] + ["None"]

# `index` is the image number -- the n in CLEVR_<split>_<n>.png. CLEVR asks
# several counting questions per image, so it repeats across rows; the pair
# (index, question) is what identifies a row.
TABLE_COLUMNS = [
    "index", "question", "ground_truth",
    "guessed_answer_raw", "guessed_answer_processed",
    "confidence", "confidence_mean", "confidence_min", "confidence_std",
]


def answer_text(text):
    """Strip the echoed prompt off a BLIP-2 response, leaving just the answer."""
    return text.rsplit("Answer:", 1)[-1].strip()


def parse_count(text):
    """Turn a raw BLIP-2 answer into the value scored against ground truth.

    Three outcomes, kept distinct because they mean different things:
      int              -- a count was stated, as digits or as a number word
      0                -- the model answered the literal string "None", i.e. it
                          asserted there are none, which is a real count of zero
      OUT_OF_DICT      -- anything else; the model said something off-vocabulary
                          rather than declining, so it is not a zero
    """
    answer = answer_text(text).lower().strip(" .!,")

    digits = re.search(r"\d+", answer)
    if digits:
        return int(digits.group())

    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", answer):
            return value

    if answer == "none":
        return 0

    return OUT_OF_DICT


def candidate_token_ids(tokenizer):
    """First-token id of each candidate answer, in CANDIDATE_ANSWERS order.

    Confidence is read off the first generation step, so only the leading token
    of each candidate is comparable. For OPT " 0".." 10" and " None" are each a
    single token, so this is exact rather than a truncation in practice.
    """
    return [tokenizer(f" {a}", add_special_tokens=False).input_ids[0]
            for a in CANDIDATE_ANSWERS]


@lru_cache(maxsize=2)
def _counting_questions(clevr_root, split):
    """Every question whose program ends in `count`, ordered by question index.

    Cached because the question file is ~150 MB and both loaders below read it.
    """
    questions_path = Path(clevr_root) / "questions" / f"CLEVR_{split}_questions.json"
    with open(questions_path) as f:
        questions = json.load(f)["questions"]

    counting = [q for q in questions if q["program"][-1]["function"] == "count"]
    return tuple(sorted(counting, key=lambda q: q["question_index"]))


def load_questions(clevr_root, split="val"):
    """Map question index -> (image filename, question text, image number).

    The image number is the digits in CLEVR_<split>_<n>.png, which is what the
    table reports as `index`. Still keyed by question index internally, since
    that is what orders the questions and keeps them distinct.
    """
    return {
        q["question_index"]: (q["image_filename"], q["question"], q["image_index"])
        for q in _counting_questions(str(clevr_root), split)
    }


def load_ground_truth(clevr_root, split="val"):
    """Map question index -> true count, from the same annotations."""
    return {
        q["question_index"]: int(q["answer"])
        for q in _counting_questions(str(clevr_root), split)
    }


@torch.no_grad()
def predict_batch(runner, image_paths, questions, max_new_tokens=10):
    """Answer one question per image, batched.

    Returns (text, confidence, cand_mean, cand_min, cand_std) per row.

    Prompts vary in length now, so the batch has to be padded. The tokenizer is
    left-padded (see `evaluate`) because generation must continue from the real
    final token of each prompt, not from a run of padding.

    Confidence is BLIP-2's own softmax over the language-model logits at the
    first generated step -- the step that carries the count -- read at the token
    the model actually emitted. The other three are mean, min and std of that
    same distribution restricted to CANDIDATE_ANSWERS: together they say how
    much mass the answer space got at all, what the model liked least, and how
    peaked it was across the answers it could have given.
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

    texts = runner.processor.batch_decode(output.sequences, skip_special_tokens=True)

    probs = output.scores[0].float().softmax(dim=-1)                  # (batch, vocab)
    first_ids = output.sequences[:, -len(output.scores)]              # (batch,)
    confidence = probs.gather(1, first_ids[:, None]).squeeze(1)       # (batch,)
    candidates = probs[:, candidate_token_ids(runner.processor.tokenizer)]  # (batch, 12)

    return list(zip(
        texts,
        confidence.tolist(),
        candidates.mean(dim=-1).tolist(),
        candidates.min(dim=-1).values.tolist(),
        candidates.std(dim=-1, unbiased=False).tolist(),
    ))


def evaluate(clevr_root=DEFAULT_CLEVR_ROOT, split="val",
             model_name="Salesforce/blip2-opt-2.7b", batch_size=16, limit=None,
             output_path=None, report_percent=1):
    clevr_root = Path(clevr_root)
    questions = load_questions(clevr_root, split)
    truth = load_ground_truth(clevr_root, split)

    image_dir = clevr_root / "images" / split
    indices = sorted(questions)
    if limit:
        indices = indices[:limit]

    images_used = len({questions[qi][0] for qi in indices})
    print(f"Evaluating {len(indices)} counting questions over {images_used} "
          f"{split} images with {model_name}\n")

    runner = Blip2Runner(model_name=model_name)
    runner.processor.tokenizer.padding_side = "left"

    records = []
    correct = 0
    start = time.time()

    total = len(indices)
    step = max(1, total * report_percent // 100)  # log once per report_percent of the data
    next_report = step

    for i in range(0, total, batch_size):
        batch = indices[i:i + batch_size]
        image_paths = [image_dir / questions[qi][0] for qi in batch]
        results = predict_batch(runner, image_paths, [questions[qi][1] for qi in batch])

        for qi, (text, confidence, cand_mean, cand_min, cand_std) in zip(batch, results):
            image_name, question, image_number = questions[qi]
            processed = parse_count(text)
            expected = truth[qi]
            is_correct = processed == expected
            correct += is_correct
            records.append({
                "index": image_number,
                "image": image_name,
                "question": question,
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
                  f"  ({rate:.1f} q/s, ~{remaining / 60:.1f} min left)")
            print(f"      latest Q  : {latest['question']}")
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
        print(f"\nPer-question table written to {output_path}")

    return records


def write_table(records, output_path):
    """Write the per-question table as CSV (or JSON if the path says .json)."""
    output_path = Path(output_path)
    rows = [{c: r[c] for c in TABLE_COLUMNS} for r in records]

    if output_path.suffix == ".json":
        output_path.write_text(json.dumps(rows, indent=2))
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def report(records, elapsed=None):
    """Print overall accuracy plus the breakdowns that show how it fails."""
    total = len(records)
    correct = sum(r["correct"] for r in records)
    counted = [r for r in records if r["guessed_answer_processed"] != OUT_OF_DICT]
    unparsed = total - len(counted)

    print("\n" + "=" * 46)
    print(f"{'RESULTS':^46}")
    print("=" * 46)
    print(f"Questions asked  : {total}")
    print(f"Images covered   : {len({r['image'] for r in records})}")
    print(f"Correct          : {correct}")
    print(f"Accuracy         : {correct / total:.2%}")

    if counted:
        mae = sum(abs(r["guessed_answer_processed"] - r["ground_truth"]) for r in counted) / len(counted)
        print(f"Mean abs. error  : {mae:.2f}  (over {len(counted)} stated counts)")
    print(f"Out of dictionary: {unparsed}  ({unparsed / total:.2%}, never correct)")
    print(f"Mean confidence  : {sum(r['confidence'] for r in records) / total:.3f}")
    if elapsed:
        print(f"Elapsed          : {elapsed / 60:.1f} min ({total / elapsed:.1f} q/s)")

    print("\nWhat the model predicted:")
    for value, n in Counter(r["guessed_answer_processed"] for r in records).most_common():
        print(f"  {str(value):>17} : {n:6d}  ({n / total:.1%})")

    if unparsed:
        print("\nMost common out-of-dictionary answers:")
        texts = Counter(r["guessed_answer_raw"] for r in records
                        if r["guessed_answer_processed"] == OUT_OF_DICT)
        for text, n in texts.most_common(10):
            shown = text if len(text) <= 60 else text[:57] + "..."
            print(f"  {n:6d}  ({n / unparsed:5.1%} of them)  {shown!r}")

    print("\nAccuracy by true count:")
    by_truth = defaultdict(list)
    for r in records:
        by_truth[r["ground_truth"]].append(r["correct"])
    for expected in sorted(by_truth):
        hits = by_truth[expected]
        print(f"  count {expected:>2} : {sum(hits):5d}/{len(hits):<5d} ({sum(hits) / len(hits):.1%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clevr-root", default=str(DEFAULT_CLEVR_ROOT), help="path to CLEVR_v1.0")
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="split with answered questions")
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--output", default=None,
                        help="write the per-question table here (.csv, or .json for JSON)")
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
