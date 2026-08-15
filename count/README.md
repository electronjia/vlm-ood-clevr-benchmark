# Counting experiments

Zero-shot counting evaluations of BLIP-2 and LLaVA-NeXT across four datasets:
CLEVR (synthetic), VQAv2 and TallyQA (natural photographs). Every run writes a
per-question table so results can be re-analysed without re-running the model.

All BLIP-2 numbers here come from `Salesforce/blip2-opt-2.7b` — the plain
pretrained checkpoint, **not** `blip2-opt-2.7b-vqa`. It has seen no VQA-style
supervision at all, only caption-based pretraining, so VQAv2 and TallyQA are
both uncontaminated probes for it. LLaVA numbers come from
`llava-hf/llava-v1.6-mistral-7b-hf`.

## Results

| # | Experiment | Model | Questions | Accuracy | MAE | Table |
|---|---|---|---|---|---|---|
| 1 | CLEVR val, counting questions | BLIP-2 | 35,422 | **17.2%** | 1.69 | [count_validation_results.csv](count_validation_results.csv) |
| 2 | CLEVR val, counting questions | LLaVA-NeXT-7B | 35,422 | **30.7%** | 1.01 | [llava_counting.csv](llava_counting.csv) |
| 3 | CLEVR, total objects per image | BLIP-2 | 15,000 | **6.5%** | 3.46 | [blip2_generic_counting.csv](blip2_generic_counting.csv) |
| 4 | VQAv2 val, "how many" subset | BLIP-2 | 23,315 | **31.5%** (37.9% soft) | — | [blip2_vqa2_counting.csv](blip2_vqa2_counting.csv) |
| 5a | TallyQA test, simple | BLIP-2 | 22,991 | **44.6%** | 0.99 | [blip2_tallyqa_simple.csv](blip2_tallyqa_simple.csv) |
| 5b | TallyQA test, complex | BLIP-2 | 15,598 | **28.4%** | 1.21 | [blip2_tallyqa_complex.csv](blip2_tallyqa_complex.csv) |

Experiments 1 and 2 run the identical question set through both models, so those
two rows are directly comparable. The others are not comparable to each other —
different answer distributions make the accuracy figure mean different things.

Full printed summaries live beside the tables as `*_summary.txt` for the runs
that produced them (4, 5a, 5b).

## The finding that runs through all of them

**Every model collapses onto a single modal answer, and accuracy is largely a
measure of how often the true count happens to equal that mode.**

| Run | Modal prediction | Share of all answers | Accuracy at the mode | Accuracy elsewhere |
|---|---|---|---|---|
| 1 · CLEVR BLIP-2 | `3` | 59.6% | 61% (truth=3) | ≤2%, except 35% at truth=0 |
| 2 · CLEVR LLaVA | `2` | 54.8% | 69% (truth=2) | 30% at 1, 22% at 3, ≤17% elsewhere |
| 3 · CLEVR totals | `3` | 87.7% | 51% (truth=3) | 0% at every other count |
| 4 · VQAv2 | `0` | 38.7% | 72% (truth=0) | ≤48%, collapsing past 4 |
| 5b · TallyQA complex | `0` | 64.2% | 70% (truth=0) | ≤15% |

Experiment 3 is the cleanest demonstration because its ground truth is nearly
uniform over 3–10: BLIP-2 answers "3" to 87.7% of images and scores 0% on all
1,925 images with 4 objects, all 1,843 with 5, and so on. Its 6.5% accuracy is
entirely the 1,898 images that happen to contain 3 objects.

Nothing here counts past about 4. Accuracy at true counts ≥ 5 is under 10% in
every run and exactly 0% in most.

### Two caveats that inflate the headline numbers

**`"None"` is scored as zero.** [`parse_count`](eval_blip2_counting.py#L54-L77)
deliberately maps the literal answer `"None"` to 0, treating it as an assertion
that there are none rather than a refusal. That is defensible in isolation, but
BLIP-2 emits it constantly: on TallyQA complex, 10,010 of the 10,016
zero-predictions are `None`/`None.`/`none` — 64% of all answers to that split.
The 70.4% accuracy on true-count-0 questions is therefore mostly a default
firing on a zero-heavy split, not evidence of counting. Any zero-accuracy figure
should be read with this in mind.

**The TallyQA simple/complex gap is confounded.** The +16.2 point simple-over-
complex gap is real but is not purely attribute binding, because the two splits
have very different answer distributions: true count 0 is 2.8% of simple
questions (637/22,991) and 27.8% of complex ones (4,335/15,598). Since BLIP-2's
behaviour on zeros differs sharply from its behaviour elsewhere, part of the gap
is composition rather than difficulty. Comparing the splits at matched true
counts would separate the two.

Separately, the VQAv2 MAE of 4919 in its summary file is a parsing artifact, not
a result: `parse_count` takes the first run of digits in the answer, so a stray
year or long number in a prose response dominates the mean. The accuracy figures
are unaffected.

## What each experiment does

### 1 & 2 — CLEVR counting questions

[`eval_blip2_counting.py`](eval_blip2_counting.py) and
[`eval_llava_counting.py`](eval_llava_counting.py)

Selects every CLEVR val question whose program ends in the `count` function
(35,422 of them over 14,251 images) and asks each image the questions CLEVR
actually poses about it — "What number of other objects are the same size as
the purple shiny object?" — rather than a generic prompt. These are compositional
counting questions: the model must filter by attribute before enumerating.

The LLaVA script imports question loading, parsing, table writing and reporting
from the BLIP-2 one so the two cannot drift apart. Two things differ and are
handled in [`eval_llava_counting.py`](eval_llava_counting.py#L38-L75): LLaVA is
instruction tuned and answers in prose, so its prompt appends "Answer with a
single number"; and its Mistral tokenizer emits digits as standalone tokens, so
confidence is read at the first generated step that states a count rather than
at step 0.

LLaVA never went off-vocabulary (0 out-of-dictionary answers vs BLIP-2's 753)
and is far more confident (mean 0.437 vs 0.091) — but note the two confidence
figures are not strictly comparable, since Mistral tokenizes multi-digit numbers
digit by digit and so cannot separate "10" from "1" in the candidate statistics.

### 3 — CLEVR total object count

[blip2_generic_counting.csv](blip2_generic_counting.csv)

One generic question per image — "How many objects are there?" — over 15,000
CLEVR images, with the scene's full object count as ground truth. No attribute
filtering, so this isolates raw enumeration from binding. It is the weakest
result of the set.

> The script that produced this table, `eval_blip2_total_count.py`, is no longer
> in the working tree. The CSV predates its removal; the run is not currently
> reproducible from this directory.

### 4 — VQAv2 "how many" subset

[blip2_vqa2_counting.csv](blip2_vqa2_counting.csv) ·
[summary](blip2_vqa2_counting_summary.txt)

23,315 counting questions over 15,875 val2014 images, scored both by exact match
(31.5%) and by the official VQA soft metric over the 10 human answers (37.9%).
Because the checkpoint has no VQA fine-tuning, this is a fair zero-shot probe —
but VQAv2's answers cluster at 1–3, which makes it a weak counting benchmark: a
model that only ever says "2" does respectably.

Requires filtering, unlike TallyQA, because VQAv2's `number` answer type also
contains clock readings and ages.

> The script that produced this table, `eval_blip2_vqa2.py`, is also no longer
> in the working tree.

### 5 — TallyQA

[`eval_blip2_tallyqa.py`](eval_blip2_tallyqa.py)

The main natural-image experiment. Every TallyQA question is a counting question
by construction, so nothing has to be filtered out. Its value here is the
simple/complex split:

* **simple** — imported from Visual Genome, naming one object class
  ("How many people are there?")
* **complex** — crowd-written, with a relation or attribute in the way
  ("How many stop signs line the road behind the cow?")

Both require the same enumeration, so the gap between them is meant to isolate
attribute binding — the same thing CLEVR's filtered questions measure, but on
real photographs. See the caveat above on reading that gap.

TallyQA is a deliberate choice for an image-in-distribution probe: its COCO/VG
imagery was seen by BLIP-2 during caption pretraining (no visual domain shift),
while its counting answers were never training targets for any BLIP-2 stage (no
contamination). Scope that claim to the image axis — its question phrasing and
answer distribution are intentionally shifted relative to VQAv2.

Test-Simple and Test-Complex are Visual Genome images only; the COCO images
appear only in `train.json`, which carries no `issimple` annotation.

## Code layout

| File | Role |
|---|---|
| [blip2.py](blip2.py) | `Blip2Runner` — model loading, `answer()`, `caption()`. Also a standalone CLI for running BLIP-2 on one image or a directory. |
| [llava_next.py](llava_next.py) | `LlavaNextRunner` and `build_prompt`, which renders the checkpoint's own chat template. |
| [eval_blip2_counting.py](eval_blip2_counting.py) | The CLEVR evaluation **and** the shared harness: `parse_count`, `answer_text`, `report`, `write_table`, `OUT_OF_DICT`. Every other eval imports from it. |
| [eval_llava_counting.py](eval_llava_counting.py) | CLEVR with LLaVA-NeXT, reusing the above. |
| [eval_blip2_tallyqa.py](eval_blip2_tallyqa.py) | TallyQA, reusing the above and adding the simple/complex breakdown. |
| [analyze_results.py](analyze_results.py) | Post-hoc analysis of any written table: per-value accuracy, confidence and confidence spread, grouped by truth (recall) or by prediction (precision). Prints a table and optionally writes plots. |

Note that `analyze_results.py` imports one constant from
`eval_blip2_counting.py` and therefore pays a full torch + transformers import
to run a pure-CSV analysis.

### Table columns

Each row is one question; `index` is the image id and repeats across rows, so
`(index, question)` identifies a row.

| Column | Meaning |
|---|---|
| `ground_truth` | True count from the dataset annotations |
| `guessed_answer_raw` | The model's answer text, prompt echo stripped |
| `guessed_answer_processed` | Parsed to an int, or `out of dictionary` |
| `confidence` | Softmax probability the model put on the token it actually emitted, at the step carrying the count |
| `confidence_mean` / `_min` / `_std` | Mean, min and std of that same distribution restricted to the candidate answers (0–10 or 0–15, plus "None") — how much mass the answer space got, what the model liked least, how peaked it was |

TallyQA tables add an `is_simple` column.

## Running

Activate the venv at the repo root first:

```bash
source ../.venv/bin/activate
```

Data is expected at `../data/raw/CLEVR_v1.0`, `../TallyQA` (VG images plus
`test.json`) and `../VQAv2`.

```bash
# CLEVR — BLIP-2 and LLaVA over the same questions
python eval_blip2_counting.py --output count_validation_results.csv
python eval_llava_counting.py  --output llava_counting.csv     # batch 4: ~2900 tokens/image

# TallyQA — both halves of the split
python eval_blip2_tallyqa.py --group simple  --output blip2_tallyqa_simple.csv
python eval_blip2_tallyqa.py --group complex --output blip2_tallyqa_complex.csv

# Post-hoc analysis of any table
python analyze_results.py blip2_tallyqa_simple.csv --by both --plot-dir .
```

Useful flags: `--limit N` for a quick smoke test, and on the TallyQA script
`--image-fraction 0.1 --seed 0` to sample images while keeping every question
about each one. TallyQA writes its summary to `<output stem>_summary.txt`
automatically; the CLEVR scripts print theirs to stdout only.

Throughput observed on one L4: BLIP-2 ~30 q/s (TallyQA complex, 15.6K questions
in 8.7 min). LLaVA-NeXT is substantially slower and needs the smaller batch.
