"""Evaluate LLaVA-NeXT-7B counting accuracy over the CLEVR counting questions.

The counterpart to eval_blip2_counting.py: same questions, same scoring, same
per-question table, so the two models' numbers are directly comparable. Question
loading, count parsing, table writing and reporting are imported from that module
rather than copied, so the metrics cannot drift apart between the two scripts.

Two things about LLaVA-NeXT differ from BLIP-2 and are handled here:

  * It is instruction tuned and answers in prose ("There are 3 cubes."), so the
    prompt asks for a bare number and generation is decoded without the prompt
    echo instead of being split on "Answer:".
  * Its Mistral tokenizer emits digits as standalone tokens preceded by a "_"
    word-start token, so generation step 0 carries no count at all. Confidence is
    therefore read at the first generated step that states a count -- see
    `count_step` -- not at step 0 as with BLIP-2's OPT decoder.

`--output` writes a per-question table (see TABLE_COLUMNS) for post-analysis.
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from eval_blip2_counting import (
    DEFAULT_CLEVR_ROOT,
    load_ground_truth,
    load_questions,
    parse_count,
    report,
    write_table,
)
from llava_next import DEFAULT_MODEL, LlavaNextRunner, build_prompt

# LLaVA-NeXT will otherwise answer in a sentence. Asking for the bare number keeps
# the answer parseable and puts the count in the first content token.
ANSWER_INSTRUCTION = "Answer with a single number."

# The answer space CLEVR counting questions can take, plus the "no answer" case.
# Mistral tokenizes multi-digit numbers digit by digit, so "10" and "1" share a
# first token: the candidate distribution below cannot tell those two apart.
CANDIDATE_ANSWERS = [str(n) for n in range(11)] + ["None"]


def candidate_token_ids(tokenizer):
    """First content-token id of each candidate answer, in CANDIDATE_ANSWERS order.

    Confidence is read at a single generation step, so only one token per
    candidate is comparable. Mistral prefixes a bare string with a word-start
    token that decodes to nothing, which is never what the model emits at the
    step that carries the count, so those leading blanks are dropped.
    """
    ids = []
    for answer in CANDIDATE_ANSWERS:
        tokens = tokenizer(answer, add_special_tokens=False).input_ids
        content = [t for t in tokens if tokenizer.decode([t]).strip()]
        ids.append(content[0])
    return ids


def count_step(tokenizer, token_ids):
    """Index of the first generated token that states a count, else 0.

    A count is a digit, or the literal "None" (a real zero, per `parse_count`).
    Falling back to step 0 keeps a confidence for off-vocabulary answers, which
    have no count token to point at; those rows score as OUT_OF_DICT anyway.
    """
    for step, token_id in enumerate(token_ids):
        piece = tokenizer.decode([token_id]).strip().lower()
        if any(c.isdigit() for c in piece) or piece == "none":
            return step
    return 0


@torch.no_grad()
def predict_batch(runner, image_paths, questions, max_new_tokens=16):
    """Answer one question per image, batched.

    Returns (text, confidence, cand_mean, cand_min, cand_std) per row.

    Prompts vary in length, so the batch has to be padded. The tokenizer is
    left-padded (see `evaluate`) because generation must continue from the real
    final token of each prompt, not from a run of padding.

    Confidence is the model's own softmax over the language-model logits at the
    step identified by `count_step` -- the step that carries the count -- read at
    the token the model actually emitted. The other three are mean, min and std of
    that same distribution restricted to CANDIDATE_ANSWERS: together they say how
    much mass the answer space got at all, what the model liked least, and how
    peaked it was across the answers it could have given.
    """
    images = [Image.open(p).convert("RGB") for p in image_paths]
    prompts = [build_prompt(runner.processor, f"{q} {ANSWER_INSTRUCTION}") for q in questions]

    inputs = runner.processor(
        images=images, text=prompts, padding=True, add_special_tokens=False,
        return_tensors="pt",
    ).to(runner.device, runner.dtype)
    output = runner.model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        output_scores=True, return_dict_in_generate=True,
    )

    tokenizer = runner.processor.tokenizer
    generated = output.sequences[:, -len(output.scores):]              # (batch, steps)
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

    probs = torch.stack(output.scores, dim=1).float().softmax(dim=-1)  # (batch, steps, vocab)
    steps = [count_step(tokenizer, row.tolist()) for row in generated]
    rows = torch.arange(len(steps))
    step_probs = probs[rows, torch.tensor(steps)]                      # (batch, vocab)

    emitted = generated[rows, torch.tensor(steps)]                     # (batch,)
    confidence = step_probs.gather(1, emitted[:, None]).squeeze(1)     # (batch,)
    candidates = step_probs[:, candidate_token_ids(tokenizer)]         # (batch, 12)

    return list(zip(
        [t.strip() for t in texts],
        confidence.tolist(),
        candidates.mean(dim=-1).tolist(),
        candidates.min(dim=-1).values.tolist(),
        candidates.std(dim=-1, unbiased=False).tolist(),
    ))


def evaluate(clevr_root=DEFAULT_CLEVR_ROOT, split="val",
             model_name=DEFAULT_MODEL, batch_size=4, limit=None,
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

    runner = LlavaNextRunner(model_name=model_name)
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
                "guessed_answer_raw": text,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clevr-root", default=str(DEFAULT_CLEVR_ROOT), help="path to CLEVR_v1.0")
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="split with answered questions")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="smaller than BLIP-2's: LLaVA-NeXT spends ~2900 tokens per image")
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
