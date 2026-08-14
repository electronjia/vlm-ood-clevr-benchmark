import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Blip2Runner:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None, dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def answer(self, image, question, max_new_tokens=30):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        prompt = f"Question: {question} Answer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, self.dtype
        )

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()

    @torch.no_grad()
    def caption(self, image, max_new_tokens=30):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.dtype)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()


def test_on_directory(image_dir, question, model_name="Salesforce/blip2-opt-2.7b", max_new_tokens=30):
    """Run the same VQA question against every image in a directory. Basic sanity tester."""
    image_dir = Path(image_dir)
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"no images found in {image_dir}")

    runner = Blip2Runner(model_name=model_name)

    results = {}
    for path in image_paths:
        answer = runner.answer(str(path), question, max_new_tokens=max_new_tokens)
        results[path.name] = answer
        print(f"{path.name}: {answer}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run BLIP-2 on an image (or directory of images)")
    parser.add_argument("image", help="path to an image file, or a directory when using --question with --dir")
    parser.add_argument("--question", "-q", default=None, help="VQA question; omit to caption instead")
    parser.add_argument("--dir", action="store_true", help="treat 'image' as a directory and test every image in it")
    parser.add_argument("--model", default="Salesforce/blip2-opt-2.7b", help="HF model name")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    if args.dir:
        if not args.question:
            raise SystemExit("--dir requires --question")
        test_on_directory(args.image, args.question, model_name=args.model, max_new_tokens=args.max_new_tokens)
        return

    runner = Blip2Runner(model_name=args.model)

    if args.question:
        result = runner.answer(args.image, args.question, max_new_tokens=args.max_new_tokens)
    else:
        result = runner.caption(args.image, max_new_tokens=args.max_new_tokens)

    print(result)


if __name__ == "__main__":
    main()
