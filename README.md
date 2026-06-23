# CLEVR VQA OOD Robustness

Benchmarking the robustness of vision-language models on out-of-distribution Visual Question Answering scenes.

## Overview

This project studies how pretrained vision-language models (VLMs) behave when Visual Question Answering (VQA) inputs move away from the distributions they are likely to have seen during training. We use the CLEVR v1.0 dataset to create controlled synthetic scenes and evaluate how model performance changes under out-of-distribution (OOD) shifts such as unusual object-color pairings, atypical positions, new orientations, and novel combinations of visual attributes.

The main goal is to quantify where VLMs fail under distribution shift and whether zero-shot prompting, few-shot prompting, or parameter-efficient fine-tuning can improve robustness.

## Research Questions

1. **Viewpoint generalization**
   How does accuracy degrade when camera azimuth or elevation changes from canonical viewpoints?

2. **Binding under scene complexity**
   How well do models bind the correct attributes to the correct objects as object count and shared attributes increase?

3. **Compositional novelty**
   Do models answer correctly for rare or unseen attribute-object combinations, or do they rely on learned visual priors?

## Models

We plan to evaluate the following pretrained VLMs:

* BLIP-2
* LLaVA-NeXT-7B
* Qwen2.5-VL-7B-Instruct

## Dataset

We use **CLEVR v1.0**, a synthetic 3-D visual reasoning dataset with ground-truth annotations for object attributes such as color, shape, material, size, position, and spatial relationships.

The dataset will be partitioned into controlled in-distribution and out-of-distribution splits. Example OOD shifts include:

* Held-out color-object combinations
* Unusual object positions
* New camera viewpoints or orientations
* Combined shifts such as color + position or color + orientation
* Increased scene complexity through more objects or overlapping attributes

## Experiment Settings

We compare model performance across three prediction settings:

### 1. Zero-shot evaluation

Models answer CLEVR questions without task-specific examples or fine-tuning.

### 2. Few-shot prompting

Models receive a small number of example image-question-answer pairs in the prompt to test whether demonstrations improve OOD robustness.

### 3. Fine-tuning

As a stretch goal, models will be fine-tuned using parameter-efficient methods such as QLoRA and evaluated on held-out CLEVR categories to test generalization.

## Evaluation Metrics

We will evaluate performance using:

* Soft semantic answer matching
* Exact match accuracy
* ID-to-OOD performance drop
* Hallucination rate
* Object, color, position, orientation, and relation correctness
* BLEU/ROUGE for caption-style outputs, if applicable
* Peak GPU memory usage
* Inference latency

A manual rubric will also be applied to sampled outputs to identify common failure modes such as attribute swaps, incorrect spatial relations, and unsupported hallucinations.

## Planned Repository Structure

```text
clevr-vqa-ood-robustness/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
│   └── prototype.ipynb
├── src/
│   ├── data/
│   │   ├── build_splits.py
│   │   └── clevr_loader.py
│   ├── models/
│   │   ├── blip2_runner.py
│   │   ├── llava_runner.py
│   │   └── qwen_runner.py
│   ├── prompting/
│   │   ├── zero_shot.py
│   │   └── few_shot.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── rubric.py
│   └── utils/
├── results/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
├── scripts/
│   ├── run_zero_shot.sh
│   ├── run_few_shot.sh
│   └── run_eval.sh
├── requirements.txt
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/clevr-vqa-ood-robustness.git
cd clevr-vqa-ood-robustness
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Download CLEVR v1.0 from the official CLEVR website and place the files under:

```text
data/raw/
```

Then create the project-specific ID/OOD splits:

```bash
python src/data/build_splits.py \
  --input_dir data/raw \
  --output_dir data/splits
```

## Running Experiments

Run zero-shot evaluation:

```bash
bash scripts/run_zero_shot.sh
```

Run few-shot evaluation:

```bash
bash scripts/run_few_shot.sh
```

Run metric computation:

```bash
bash scripts/run_eval.sh
```

## Expected Outputs

The project will produce:

* A curated CLEVR ID/OOD benchmark split
* A unified evaluation harness for multiple VLMs
* Zero-shot and few-shot performance comparisons
* ID-to-OOD degradation analysis
* Error analysis for hallucination, attribute binding, and spatial reasoning failures
* A reproducible prototype notebook or runnable codebase
* Final report and presentation materials

## Hardware

Planned compute resources:

* NVIDIA RTX 3080 Ti, 12 GB VRAM
* NVIDIA RTX 4090, 16 GB VRAM

## References

* Agrawal et al., *VQA: Visual Question Answering*
* Johnson et al., *CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning*
* Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*
* LLaVA Team, *LLaVA-NeXT*
* Bai et al., *Qwen2.5-VL Technical Report*
  ::: 
