# BLIP-2 CLEVR Failure Analysis — Reproducible Pipeline

This documents the exact order to reproduce every result, from raw CLEVR data to
the final figures. Each stage is an independent script so you can run or verify
them one at a time (important given the GPU / library constraints).

## Contribution scope

This pipeline covers the **zero-shot baseline (BLIP-2, Qwen2.5-VL)** and the
**diagnostic analysis of BLIP-2's failure** (Q-Former compression analysis and
linear probing across the vision encoder, Q-Former, and language-model layers).

## Environment

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`.

**Windows / conda note.** `numpy`'s SVD and some BLAS calls can hang under an
MKL/OpenMP conflict. Every script that does linear algebra sets these at the top
*before* importing numpy — keep them:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
```

## Expected directory layout

```
clevr/
  questions/CLEVR_val_questions.json
  questions/CLEVR_train_questions.json
  scenes/CLEVR_val_scenes.json
  scenes/CLEVR_train_scenes.json
  images/val/CLEVR_val_XXXXXX.png
results/            # all outputs land here
```

---

## Pipeline stages (run in this order)

### Stage 1 — Build the dataframes
`build_dataframes.py`

Parses CLEVR questions + scenes into one row per (image, question) with scene
attributes and question metadata. This is the foundation every later stage reads.

```bash
python build_dataframes.py --split val
python build_dataframes.py --split train      # only needed for fine-tuning
```
**Output:** `val_dataframe.csv`, `train_dataframe.csv`
(columns: image_filename, image_index, split, num_objects, colors, shapes, sizes,
materials, sphere_present, red_present, gray_present, question_index,
question_type, question_complexity, question, answer)

### Stage 2 — Zero-shot baseline (BLIP-2)  ·  GPU
`run_blip2.py`

Runs BLIP-2 (OPT-2.7B) on every CLEVR val question, scores with soft-match.

```bash
python run_blip2.py --csv val_dataframe.csv
```
**Output:** `blip2_clevr_results_final.csv` (adds `prediction`, `correct`)

### Stage 3 — Aggregate results
`aggregate_results.py`

Groups the baseline predictions into the summary tables used in the report.

```bash
python aggregate_results.py --results blip2_clevr_results_final.csv
```
**Output:** `results_overall.csv`, `results_by_question_type.csv`,
`results_by_complexity.csv`, `results_by_type_complexity.csv`

### Stage 4 — Failure analysis (count collapse)  ·  CPU
`failure_analysis.py`

Parses count predictions, builds the true-vs-predicted distribution and the
confusion matrix showing the collapse onto "3".

```bash
python failure_analysis.py --csv blip2_clevr_results_final.csv --out results/
```
**Output:** `count_distribution_comparison.png`, `count_confusion_matrix.png`,
`complexity_paradox.png`

### Stage 5 — Q-Former compression analysis (cosine + effective rank)  ·  GPU then CPU
`cosine_analysis.py` (+ `inspect_qformer.py` for the forward hooks)

Two steps. Step 1 runs BLIP-2 to capture the 32 Q-Former tokens per image and
saves them. Step 2 computes mean cosine similarity and effective rank.

```bash
# step 1: extract Q-Former tensors (GPU)
python inspect_qformer.py --csv val_dataframe.csv --save-tensors
# step 2: cosine similarity + effective rank (CPU)
python cosine_analysis.py --all
```
**Output:** `qformer_metrics_all.csv` (per-image `mean_sim`, `eff_rank`),
plus the aggregate `qformer_overall.csv`, `qformer_by_question_type.csv`,
`qformer_by_correct.csv`, `qformer_by_complexity.csv`.

> Note: this stage originally had to be run inside the notebook due to the
> conda LAPACK hang.

### Stage 6 — Q-Former probing (does compression lose count?)  ·  GPU then CPU
`probing_qformer.py`  (two sub-stages)

Predicts total object count from the vision encoder vs. the Q-Former output.

```bash
# quick test first
python probing_qformer.py extract --csv val_dataframe.csv
python probing_qformer.py train
# then the full run
python probing_qformer.py extract --csv val_dataframe.csv
python probing_qformer.py train
```
**Output:** `probe_vision_X.npy`, `probe_qformer_X.npy`, `probe_labels_y.npy`,
and the printed exact / within-±1 accuracy vs. baseline.

### Stage 7 — LLM-depth probing (does the LLM lose count?)  ·  GPU then CPU
`probing_llm.py`  (two sub-stages)

Same probe applied at 6 depths: Q-Former → projection → OPT layers 0/8/16/last.

```bash
python probing_llm.py extract --csv val_dataframe.csv 
python probing_llm.py train
```
**Output:** per-depth `X_*.npy`, `y.npy`, and `count_depth_curve.png`.

### Stage 8 — Qwen2.5-VL baseline (comparison)  ·  GPU
`run_qwen.py`

Runs Qwen2.5-VL on CLEVR for the cross-model comparison. Batched, resumable,
with a visual-resolution cap so it's tractable.

```bash
python run_qwen.py --csv val_dataframe.csv --sample 15000 --batch-size 8
```
**Output:** `qwen_clevr_results.csv` + per-question-type accuracy.

---