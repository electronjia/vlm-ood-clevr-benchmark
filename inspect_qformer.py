"""
inspect_qformer.py — Inspect BLIP-2's vision encoder output vs Q-Former output.

BLIP-2 has a key architectural bottleneck:
    Vision encoder (ViT)  →  257 patch tokens, 1408-dim each, 1024 dim?
    Q-Former              →  32 query tokens,  768-dim each

That's the "compression": ~257 spatial patches get squeezed into 32 fixed
query vectors before the language model ever sees them. This script captures
BOTH tensors for a given image and reports their shapes, statistics, and
how much they differ — so you can investigate what the bottleneck keeps
and what it discards.

Use forward hooks: PyTorch lets you attach a function to any submodule
that fires when that module runs, capturing its input/output without
modifying the model code.

Usage:
    python inspect_qformer.py --image CLEVR_val_000000.png
    python inspect_qformer.py --image CLEVR_val_000000.png --question "Are there any other things that are the same shape as the big metallic object?"
    python inspect_qformer.py --image CLEVR_val_000000.png --save-tensors
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from config import CLEVR_IMAGES, RESULTS_DIR

BLIP2_ID = "Salesforce/blip2-opt-2.7b"


def load_blip2():
    """Load BLIP-2 (fp16, on GPU if available)."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print(f"Loading BLIP-2 ({BLIP2_ID})...")
    processor = Blip2Processor.from_pretrained(BLIP2_ID, use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_ID, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    print("  Loaded!\n")
    return model, processor


def tensor_stats(name, t):
    """Print shape + summary statistics for a tensor."""
    t = t.float()   # cast from fp16 so numpy stats don't overflow
    arr = t.detach().cpu().numpy()
    # print(f"  {name}")
    # print(f"    shape:     {tuple(arr.shape)}")
    # print(f"    dtype:     {arr.dtype}")
    # print(f"    mean:      {arr.mean():.4f}")
    # print(f"    std:       {arr.std():.4f}")
    # print(f"    min / max: {arr.min():.4f} / {arr.max():.4f}")
    # print(f"    L2 norm:   {np.linalg.norm(arr):.2f}")
    # print()
    return arr


def main():
    parser = argparse.ArgumentParser(description="Inspect BLIP-2 vision encoder vs Q-Former output")
    parser.add_argument("--image", default="CLEVR_val_000000.png", help="Image filename (in --images folder)")
    parser.add_argument("--images", default=CLEVR_IMAGES, help="Folder containing images")
    parser.add_argument("--question", default="Are there any other things that are the same shape as the big metallic object?",
                        help="Question context (BLIP-2's Q-Former is question-agnostic, "
                             "but the language model uses it)")
    parser.add_argument("--save-tensors", action="store_true",
                        help="Save the raw tensors as .npy files for further analysis")
    args = parser.parse_args()

    #  Load model and image 
    model, processor = load_blip2()

    img_path = os.path.join(args.images, args.image)
    image = Image.open(img_path).convert("RGB")
    device = next(model.parameters()).device

    #  Set up hooks to capture intermediate outputs 
    # A dict to stash whatever the hooks capture.
    captured = {}

    def make_hook(name):
        def hook(module, inputs, output):
            # Some modules return a tuple or a ModelOutput; grab the main tensor.
            if isinstance(output, tuple):
                captured[name] = output[0]
            elif hasattr(output, "last_hidden_state"):
                captured[name] = output.last_hidden_state
            else:
                captured[name] = output
        return hook

    # model.vision_model = the ViT encoder
    # model.qformer      = the Q-Former
    h1 = model.vision_model.register_forward_hook(make_hook("vision_output"))
    h2 = model.qformer.register_forward_hook(make_hook("qformer_output"))

    #  Run a forward pass 
    prompt = f"Question: {args.question} Answer:"
    inputs = processor(image, prompt, return_tensors="pt").to(device, torch.float16)

    print("Running forward pass with hooks...\n")
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    # Remove hooks so they don't fire again
    h1.remove()
    h2.remove()

    # Report 
    print("=" * 55)
    print("  BLIP-2 INTERMEDIATE OUTPUTS")
    print("=" * 55 + "\n")

    print("── Vision encoder output (before Q-Former) ──")
    vision_arr = tensor_stats("vision_model output", captured["vision_output"])

    print("── Q-Former output (the compressed representation) ──")
    qformer_arr = tensor_stats("qformer output", captured["qformer_output"])

    # ── The compression story ────────────────────────────────────
    # vision_arr shape: (batch, num_patches, vision_dim) e.g. (1, 257, 1408)
    # qformer_arr shape: (batch, num_queries, qformer_dim) e.g. (1, 32, 768)
    v_tokens = vision_arr.shape[1]
    q_tokens = qformer_arr.shape[1]
    v_dim = vision_arr.shape[2]
    q_dim = qformer_arr.shape[2]

    v_total = v_tokens * v_dim
    q_total = q_tokens * q_dim

    print("=" * 55)
    print("  COMPRESSION SUMMARY")
    print("=" * 55)
    print(f"  Vision encoder: {v_tokens} tokens × {v_dim} dims = {v_total:,} values")
    print(f"  Q-Former:       {q_tokens} tokens × {q_dim} dims = {q_total:,} values")
    print(f"  Token reduction:  {v_tokens} → {q_tokens}  ({v_tokens/q_tokens:.1f}× fewer)")
    print(f"  Total reduction:  {v_total:,} → {q_total:,}  ({v_total/q_total:.1f}× smaller)")
    print("=" * 55 + "\n")

    #  save tensors for deeper analysis 
    if args.save_tensors:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stem = args.image.replace(".png", "")
        v_path = os.path.join(RESULTS_DIR, f"{stem}_vision.npy")
        q_path = os.path.join(RESULTS_DIR, f"{stem}_qformer.npy")
        np.save(v_path, vision_arr)
        np.save(q_path, qformer_arr)
        print(f"  Saved tensors:")
        print(f"    {v_path}   (vision encoder output)")
        print(f"    {q_path}   (Q-Former output)")
        print(f"  Load them later with: np.load(path)\n")


if __name__ == "__main__":
    main()