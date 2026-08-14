import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"


def build_prompt(processor, question=None):
    """Render the model's own chat template around an image (and a question).

    LLaVA-NeXT is instruction tuned, so the wrapping matters: the Mistral 7B
    checkpoint expects "[INST] <image>\\n... [/INST]", the Vicuna one a different
    wrapping entirely. Going through the template keeps this correct for whichever
    checkpoint is loaded. The template emits its own BOS, so the tokenizer must be
    called with add_special_tokens=False.
    """
    content = "<image>\n" + (question if question else "Describe this image.")
    return processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=False,
    )


class LlavaNextRunner:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _generate(self, image, prompt, max_new_tokens):
        inputs = self.processor(
            images=image, text=prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device, self.dtype)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Decode only what was generated; the prompt is echoed back in output_ids.
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    @torch.no_grad()
    def answer(self, image, question, max_new_tokens=30):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        return self._generate(image, build_prompt(self.processor, question), max_new_tokens)

    @torch.no_grad()
    def caption(self, image, max_new_tokens=30):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        return self._generate(image, build_prompt(self.processor), max_new_tokens)


def test_on_directory(image_dir, question, model_name=DEFAULT_MODEL, max_new_tokens=30):
    """Run the same VQA question against every image in a directory. Basic sanity tester."""
    image_dir = Path(image_dir)
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"no images found in {image_dir}")

    runner = LlavaNextRunner(model_name=model_name)

    results = {}
    for path in image_paths:
        answer = runner.answer(str(path), question, max_new_tokens=max_new_tokens)
        results[path.name] = answer
        print(f"{path.name}: {answer}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLaVA-NeXT-7B on an image (or directory of images)")
    parser.add_argument("image", help="path to an image file, or a directory when using --question with --dir")
    parser.add_argument("--question", "-q", default=None, help="VQA question; omit to caption instead")
    parser.add_argument("--dir", action="store_true", help="treat 'image' as a directory and test every image in it")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model name")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    if args.dir:
        if not args.question:
            raise SystemExit("--dir requires --question")
        test_on_directory(args.image, args.question, model_name=args.model, max_new_tokens=args.max_new_tokens)
        return

    runner = LlavaNextRunner(model_name=args.model)

    if args.question:
        result = runner.answer(args.image, args.question, max_new_tokens=args.max_new_tokens)
    else:
        result = runner.caption(args.image, max_new_tokens=args.max_new_tokens)

    print(result)


if __name__ == "__main__":
    main()
