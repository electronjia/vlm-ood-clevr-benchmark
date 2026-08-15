"""Evaluate BLIP-2 counting accuracy over TallyQA.

The natural-image counterpart to eval_blip2_counting.py, and the harder one:
every TallyQA question is a counting question by construction, so unlike VQA v2
nothing has to be filtered out (see eval_blip2_vqa2.py, where the `number`
answer type also holds clock readings and ages). Prompting, count parsing and
the report are imported from the CLEVR module rather than copied, so the
metrics cannot drift apart between the two.

What TallyQA adds is the simple/complex split. Simple questions are imported
from Visual Genome and name one object class ("How many people are there?");
complex ones are crowd-written and put a relation or attribute in the way
("How many stop signs line the road behind the cow?"). Both need the same
count, so the gap between them isolates attribute binding from enumeration --
the same thing CLEVR's filtered questions measure, on real photographs.
Accuracy is therefore reported per group as well as overall.

Questions come from test.json (which carries `issimple`) or train.json (which
does not, so it reports overall only). Image paths in those files are relative
to the TallyQA root: VG_100K/, VG_100K_2/ for Visual Genome, train2014/,
val2014/ for the COCO images train.json also draws on.

A run writes both of its results to disk without being asked: the per-question
table (see TABLE_COLUMNS) to `--output`, and the RESULTS summary printed at the
end to `--summary`, which defaults to <output stem>_summary.txt beside it.
"""

import argparse
import csv
import io
import json
import random
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from blip2 import Blip2Runner
from eval_blip2_counting import OUT_OF_DICT, answer_text, parse_count
from eval_blip2_counting import report as clevr_report

DEFAULT_TALLYQA_ROOT = Path(__file__).resolve().parents[1] / "TallyQA"

# Written unless --output says otherwise, next to the other result tables. The
# summary goes to <stem>_summary.txt beside it, so a run never has to be
# repeated just to recover its numbers.
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "blip2_tallyqa_counting.csv"

# The answer space TallyQA questions can take, plus the "no answer" case.
# 0-15 is the whole of it -- both splits top out at 15 -- and BLIP-2 generates
# after "Answer:", so the candidate strings carry a leading space. Each of
# " 0".." 15" and " None" is a single OPT token, which is what reading
# confidence off the first generated step requires.
CANDIDATE_ANSWERS = [str(n) for n in range(16)] + ["None"]

# `index` is the image id. TallyQA asks several questions per image, so it
# repeats across rows; the pair (index, question) is what identifies a row.
# `is_simple` is the split described in the module docstring, blank for train.
TABLE_COLUMNS = [
    "index", "question", "ground_truth", "is_simple",
    "guessed_answer_raw", "guessed_answer_processed",
    "confidence", "confidence_mean", "confidence_min", "confidence_std",
]


def candidate_token_ids(tokenizer):
    """First-token id of each candidate answer, in CANDIDATE_ANSWERS order.

    Confidence is read off the first generation step, so only the leading token
    of each candidate is comparable. For OPT " 0".." 15" and " None" are each a
    single token, so this is exact rather than a truncation in practice. Ids are
    de-duplicated in case another tokenizer splits two candidates to the same
    leading token, which would otherwise double-count that token in the stats.
    """
    ids = [tokenizer(f" {a}", add_special_tokens=False).input_ids[0]
           for a in CANDIDATE_ANSWERS]
    return list(dict.fromkeys(ids))


@lru_cache(maxsize=2)
def _questions(tallyqa_root, split):
    """Every TallyQA question in `split`, ordered by question id.

    No filtering and no dropped rows: the answers are already integers in the
    annotation file, so unlike CLEVR and VQA v2 there is nothing to parse or
    discard on the ground-truth side. Cached because train.json is ~47 MB and
    both loaders below read it.

    train.json has no `issimple` field -- the simple/complex distinction is
    only annotated on test -- so it comes back as None there and the report
    skips that breakdown.
    """
    with open(Path(tallyqa_root) / f"{split}.json") as f:
        entries = json.load(f)

    questions = tuple(sorted(
        (
            {
                "question_id": e["question_id"],
                "image": e["image"],
                "image_id": e["image_id"],
                "question": e["question"],
                "ground_truth": int(e["answer"]),
                "is_simple": e.get("issimple"),
            }
            for e in entries
        ),
        key=lambda q: q["question_id"],
    ))
    return questions


def load_questions(tallyqa_root, split="test", group="all"):
    """Map question id -> (image path, question text, image id).

    `group` keeps all questions, or only the simple or only the complex ones;
    the latter two are annotated on test alone.
    """
    questions = _questions(str(tallyqa_root), split)
    if group != "all":
        wanted = group == "simple"
        questions = [q for q in questions if q["is_simple"] == wanted]
    return {
        q["question_id"]: (q["image"], q["question"], q["image_id"])
        for q in questions
    }


def load_ground_truth(tallyqa_root, split="test", group="all"):
    """Map question id -> (true count, is_simple), from the same file."""
    questions = _questions(str(tallyqa_root), split)
    if group != "all":
        wanted = group == "simple"
        questions = [q for q in questions if q["is_simple"] == wanted]
    return {q["question_id"]: (q["ground_truth"], q["is_simple"]) for q in questions}


@torch.no_grad()
def predict_batch(runner, image_paths, questions, max_new_tokens=10):
    """Answer one question per image, batched.

    Returns (text, confidence, cand_mean, cand_min, cand_std) per row.

    Identical to eval_blip2_counting.predict_batch except for the candidate set
    the last three are computed over, which is TallyQA's answer space rather
    than CLEVR's. Prompts vary in length, so the batch has to be padded; the
    tokenizer is left-padded (see `evaluate`) because generation must continue
    from the real final token of each prompt, not from a run of padding.

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
    candidates = probs[:, candidate_token_ids(runner.processor.tokenizer)]  # (batch, 17)

    return list(zip(
        texts,
        confidence.tolist(),
        candidates.mean(dim=-1).tolist(),
        candidates.min(dim=-1).values.tolist(),
        candidates.std(dim=-1, unbiased=False).tolist(),
    ))


def sample_images(questions, fraction, seed):
    """Question ids for a random `fraction` of the images, all questions kept.

    Sampling images rather than questions keeps every question about a chosen
    image, so the per-image question mix is the one TallyQA actually has --
    what `--limit` alone cannot give, since it slices the first N question ids
    and those are ordered by image.
    """
    images = sorted({questions[qi][2] for qi in questions})
    chosen = set(random.Random(seed).sample(images, round(fraction * len(images))))
    return sorted(qi for qi in questions if questions[qi][2] in chosen)


def evaluate(tallyqa_root=DEFAULT_TALLYQA_ROOT, split="test", group="all",
             model_name="Salesforce/blip2-opt-2.7b", batch_size=16, limit=None,
             image_fraction=None, seed=0, output_path=DEFAULT_OUTPUT,
             summary_path=None, report_percent=1):
    tallyqa_root = Path(tallyqa_root)
    output_path = Path(output_path) if output_path else None
    # The summary sits beside the table by default, so a run leaves both the
    # per-question rows and the numbers computed from them on disk.
    if summary_path is None and output_path:
        summary_path = output_path.with_name(output_path.stem + "_summary.txt")

    questions = load_questions(tallyqa_root, split, group)
    truth = load_ground_truth(tallyqa_root, split, group)

    if image_fraction:
        indices = sample_images(questions, image_fraction, seed)
    else:
        indices = sorted(questions)
    if limit:
        indices = indices[:limit]

    if not indices:
        # Almost always --group simple/complex against train, which carries no
        # issimple field. Say so before spending a minute loading the model.
        raise SystemExit(
            f"no {group} questions in the {split} split"
            + (" -- the simple/complex annotation is only on test"
               if group != "all" and split != "test" else ""))

    images_used = len({questions[qi][0] for qi in indices})
    sampled = f" ({image_fraction:.0%} image sample, seed {seed})" if image_fraction else ""
    print(f"Evaluating {len(indices)} {group} counting questions over {images_used} "
          f"{split} images{sampled} with {model_name}\n")

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
        image_paths = [tallyqa_root / questions[qi][0] for qi in batch]
        results = predict_batch(runner, image_paths, [questions[qi][1] for qi in batch])

        for qi, (text, confidence, cand_mean, cand_min, cand_std) in zip(batch, results):
            image_name, question, image_id = questions[qi]
            processed = parse_count(text)
            expected, is_simple = truth[qi]
            is_correct = processed == expected
            correct += is_correct
            records.append({
                "index": image_id,
                "image": image_name,
                "question": question,
                "ground_truth": expected,
                "is_simple": is_simple,
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
    report(records, elapsed, output_path=summary_path)

    if output_path:
        write_table(records, output_path)
        print(f"\nPer-question table written to {output_path}")
    if summary_path:
        print(f"Summary written to {summary_path}")

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


def report(records, elapsed=None, output_path=None):
    """The CLEVR report, plus TallyQA's simple/complex breakdown.

    The shared part is captured from eval_blip2_counting.report rather than
    reimplemented, so overall accuracy, mean absolute error, the
    out-of-dictionary tally and the by-true-count table are computed by the
    same code that produced the CLEVR numbers and cannot drift from them.

    Every line is also written to `output_path` when given, so the summary
    survives the run rather than living only in the terminal. Returns the text
    either way.
    """
    shared = io.StringIO()
    with redirect_stdout(shared):
        clevr_report(records, elapsed)
    lines = shared.getvalue().rstrip("\n").split("\n")

    groups = defaultdict(list)
    for r in records:
        if r["is_simple"] is not None:
            groups["simple" if r["is_simple"] else "complex"].append(r)

    if groups:
        lines.append("")
        lines.append("Accuracy by question type:")
        for name in ("simple", "complex"):
            rows = groups.get(name)
            if not rows:
                continue
            hits = sum(r["correct"] for r in rows)
            unparsed = sum(r["guessed_answer_processed"] == OUT_OF_DICT for r in rows)
            confidence = sum(r["confidence"] for r in rows) / len(rows)
            lines.append(f"  {name:>7} : {hits:6d}/{len(rows):<6d} ({hits / len(rows):6.1%})"
                         f"  conf {confidence:.3f}, out of dictionary {unparsed / len(rows):.1%}")
        if len(groups) == 2:
            simple = sum(r["correct"] for r in groups["simple"]) / len(groups["simple"])
            complex_ = sum(r["correct"] for r in groups["complex"]) / len(groups["complex"])
            lines.append(f"  {'gap':>7} : {simple - complex_:+.1%}"
                         f"  (simple minus complex)")
    else:
        lines.append("")
        lines.append("Accuracy by question type: not annotated on this split.")

    text = "\n".join(lines)
    print(text)
    if output_path:
        Path(output_path).write_text(text + "\n")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tallyqa-root", default=str(DEFAULT_TALLYQA_ROOT),
                        help="path to the TallyQA directory (test.json plus the image dirs)")
    parser.add_argument("--split", default="test", choices=["test", "train"],
                        help="test carries the simple/complex annotation; train does not")
    parser.add_argument("--group", default="all", choices=["all", "simple", "complex"],
                        help="restrict to one side of the simple/complex split (test only)")
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--image-fraction", type=float, default=None,
                        help="evaluate a random fraction of the images, e.g. 0.1, "
                             "keeping every question about each one")
    parser.add_argument("--seed", type=int, default=0, help="seed for --image-fraction")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="write the per-question table here (.csv, or .json for JSON); "
                             '"" to skip it')
    parser.add_argument("--summary", default=None,
                        help="write the RESULTS summary here "
                             "(default: <output stem>_summary.txt)")
    parser.add_argument("--report-percent", type=int, default=1,
                        help="print a progress line every N%% of the dataset")
    args = parser.parse_args()

    evaluate(
        tallyqa_root=args.tallyqa_root,
        split=args.split,
        group=args.group,
        model_name=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        image_fraction=args.image_fraction,
        seed=args.seed,
        output_path=args.output,
        summary_path=args.summary,
        report_percent=args.report_percent,
    )


if __name__ == "__main__":
    main()
