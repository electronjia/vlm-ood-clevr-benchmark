"""Grad-CAM saliency maps for BLIP-2 answers.

Shows which image regions drove a particular answer, which is how we tell a
genuine miscount apart from the model ignoring the image altogether.

Two attribution sites are available:
  xattn  Q-Former cross-attention (default) -- where the query tokens that feed
         the language model actually looked. Most faithful to BLIP-2 grounding.
  vit    Classic ViT Grad-CAM on the last vision-encoder block.

IMPORTANT -- what these maps can and cannot tell you
----------------------------------------------------
For blip2-opt the Q-Former is called with only `query_embeds` (learned and
fixed) and the image embeddings; the question text never reaches it. The
forward cross-attention is therefore a function of the image alone, and
measurements on CLEVR bear that out: swapping the explained answer ("3" vs
"10") leaves the map at r=0.9998, and even swapping in an unrelated question
leaves it at r=0.99. Using the gradient alone rather than activation*gradient
does not separate them either (r=0.9996).

So treat the output as an image saliency map, not as evidence about how a
particular answer was reached. Peaks also cluster in the top image row
regardless of scene content -- the high-norm background "register" tokens ViTs
are known to develop -- so do not read an isolated background hotspot as the
model attending to that spot. For answer-specific attribution, perturbation
methods (masking objects and measuring the answer's change) are the sounder
route here.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

DEFAULT_QUESTION = "How many objects are in the image?"
# Q-Former only has cross-attention on every other layer; middle layers are the
# usual sweet spot (early = low-level, late = already abstracted).
CROSS_ATTENTION_LAYERS = (0, 2, 4, 6, 8, 10)


class Blip2GradCAM:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None, method="xattn", layer=6):
        if method not in ("xattn", "vit"):
            raise ValueError(f"method must be 'xattn' or 'vit', got {method!r}")
        if method == "xattn" and layer not in CROSS_ATTENTION_LAYERS:
            raise ValueError(f"layer {layer} has no cross-attention; pick one of {CROSS_ATTENTION_LAYERS}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method
        self.layer = layer

        self.processor = Blip2Processor.from_pretrained(model_name)
        # float32 + eager attention: fp16 backward is unstable, and the fused
        # kernels don't expose the attention probabilities we need.
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32, attn_implementation="eager"
        ).to(self.device)
        self.model.eval()

        # Freeze everything: we want gradients w.r.t. activations, not weights.
        # The graph is kept alive by making pixel_values require grad instead.
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.grid = self.model.config.vision_config.image_size // self.model.config.vision_config.patch_size
        self._captured = None
        self._capturing = False  # the hook also fires under generate(), where there is no graph
        self._register_hook()

    def _register_hook(self):
        if self.method == "xattn":
            module = self.model.qformer.encoder.layer[self.layer].crossattention.attention
        else:
            module = self.model.vision_model.encoder.layers[-1]

        def hook(_module, _inputs, output):
            if not self._capturing:
                return
            # xattn returns (context, attention_probs); the ViT block returns hidden states.
            tensor = output[1] if self.method == "xattn" else (
                output[0] if isinstance(output, (tuple, list)) else output
            )
            tensor.retain_grad()
            self._captured = tensor

        module.register_forward_hook(hook)

    def _build_inputs(self, image, question, answer):
        """Tokenize prompt+answer, masking the prompt so the loss is answer-specific."""
        prompt = f"Question: {question} Answer:"
        full = f"{prompt} {answer}"

        inputs = self.processor(images=image, text=full, return_tensors="pt").to(self.device, torch.float32)
        n_prompt = len(self.processor.tokenizer(prompt).input_ids)

        labels = inputs["input_ids"].clone()
        labels[:, :n_prompt] = -100
        return inputs, labels

    def __call__(self, image, question=DEFAULT_QUESTION, answer=None, max_new_tokens=10):
        """Return (cam, answer) where cam is a normalized grid x grid saliency map."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        if answer is None:
            answer = self.generate(image, question, max_new_tokens=max_new_tokens)

        inputs, labels = self._build_inputs(image, question, answer)
        inputs["pixel_values"].requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        self._captured = None
        self._capturing = True
        try:
            loss = self.model(**inputs, labels=labels).loss
            loss.backward()
        finally:
            self._capturing = False

        activations = self._captured
        if activations is None:
            raise RuntimeError("hooked layer never fired")
        gradients = activations.grad
        if gradients is None:
            raise RuntimeError("no gradients reached the hooked layer")

        if self.method == "xattn":
            # (1, heads, queries, tokens) -> weight attention by its gradient,
            # then pool over heads and query tokens.
            cam = (activations * gradients).clamp(min=0).mean(dim=1).mean(dim=1)[0]
        else:
            # (1, tokens, channels) -> standard Grad-CAM channel pooling.
            cam = (activations * gradients).sum(dim=-1).clamp(min=0)[0]

        cam = cam[1:]  # drop the CLS token, leaving one score per patch
        cam = cam.reshape(self.grid, self.grid).detach().float().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, answer

    @torch.no_grad()
    def generate(self, image, question=DEFAULT_QUESTION, max_new_tokens=10):
        """The model's own answer, used as the Grad-CAM target when none is given."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(
            images=image, text=f"Question: {question} Answer:", return_tensors="pt"
        ).to(self.device, torch.float32)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.rsplit("Answer:", 1)[-1].strip()


def save_overlay(image, cam, out_path, question=None, answer=None, alpha=0.5):
    """Write a side-by-side figure: original image next to the saliency overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # Upsample the coarse patch grid to full resolution for a readable overlay.
    heat = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(image)
    axes[0].set_title("input")
    axes[1].imshow(image)
    overlay = axes[1].imshow(heat, cmap="jet", alpha=alpha)
    axes[1].set_title(f"Grad-CAM (answer: {answer})" if answer else "Grad-CAM")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(overlay, ax=axes[1], fraction=0.046)

    if question:
        fig.suptitle(question)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--question", "-q", default=DEFAULT_QUESTION)
    parser.add_argument("--answer", "-a", default=None,
                        help="explain this answer instead of the model's own (e.g. the ground-truth count)")
    parser.add_argument("--output", "-o", default="gradcam.png")
    parser.add_argument("--method", default="xattn", choices=["xattn", "vit"])
    parser.add_argument("--layer", type=int, default=6, help=f"Q-Former layer, one of {CROSS_ATTENTION_LAYERS}")
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--alpha", type=float, default=0.5, help="heatmap opacity")
    args = parser.parse_args()

    cammer = Blip2GradCAM(model_name=args.model, method=args.method, layer=args.layer)
    cam, answer = cammer(args.image, args.question, args.answer)

    print(f"question : {args.question}")
    print(f"answer   : {answer}{'  (forced)' if args.answer else '  (model generated)'}")
    print(f"cam      : {cam.shape[0]}x{cam.shape[1]} grid, method={args.method}, layer={args.layer}")

    save_overlay(args.image, cam, args.output, args.question, answer, alpha=args.alpha)
    print(f"saved    : {args.output}")


if __name__ == "__main__":
    main()
