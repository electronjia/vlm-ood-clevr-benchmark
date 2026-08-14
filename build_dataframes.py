"""
build_dataframes.py — Build val_dataframe.csv and train_dataframe.csv from CLEVR.

Reads the CLEVR questions + scenes JSON and produces one row per (image, question)
with the scene attributes and question metadata used throughout the pipeline.

This code was originally developed in the exploratory notebook; it is extracted
here so the dataframes can be regenerated reproducibly.

Output columns:
    image_filename, image_index, split, num_objects,
    colors, shapes, sizes, materials,
    sphere_present, red_present, gray_present,
    question_index, question_type, question_complexity, question, answer

Usage:
    python build_dataframes.py --split val
    python build_dataframes.py --split train
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def assign_complexity(q):
    """Complexity from the length of the CLEVR functional program."""
    num_steps = len(q["program"])
    if num_steps <= 5:
        return "simple"
    elif num_steps <= 9:
        return "medium"
    else:
        return "complex"


def build_dataframe(split, clevr_root):
    clevr_root = Path(clevr_root)
    question_file = clevr_root / "questions" / f"CLEVR_{split}_questions.json"
    scene_file = clevr_root / "scenes" / f"CLEVR_{split}_scenes.json"

    print(f"Reading {question_file}")
    with open(question_file) as f:
        questions = json.load(f)["questions"]

    print(f"Reading {scene_file}")
    with open(scene_file) as f:
        scenes = json.load(f)["scenes"]

    # Group questions by image so each scene's questions are together
    questions_by_image = defaultdict(list)
    for q in questions:
        questions_by_image[q["image_index"]].append(q)

    rows = []
    for scene in scenes:
        objects = scene["objects"]
        image_index = scene["image_index"]

        colors = sorted(set(o["color"] for o in objects))
        shapes = sorted(set(o["shape"] for o in objects))
        sizes = sorted(set(o["size"] for o in objects))
        materials = sorted(set(o["material"] for o in objects))

        sphere_present = "sphere" in shapes
        red_present = "red" in colors
        gray_present = "gray" in colors

        for q in questions_by_image[image_index]:
            # question_type = the final function in the CLEVR program
            question_type = q["program"][-1]["function"]
            rows.append({
                "image_filename": scene["image_filename"],
                "image_index": image_index,
                "split": scene.get("split", split),
                "num_objects": len(objects),
                "colors": ", ".join(colors),
                "shapes": ", ".join(shapes),
                "sizes": ", ".join(sizes),
                "materials": ", ".join(materials),
                "sphere_present": sphere_present,
                "red_present": red_present,
                "gray_present": gray_present,
                "question_index": q["question_index"],
                "question_type": question_type,
                "question_complexity": assign_complexity(q),
                "question": q["question"],
                "answer": q["answer"],
            })

    df = pd.DataFrame(rows)
    out = f"{split}_dataframe.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}  —  {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  question types: {df['question_type'].nunique()}")
    print(f"  num_objects range: {df['num_objects'].min()}–{df['num_objects'].max()}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "train"])
    ap.add_argument("--clevr-root", default="clevr",
                    help="Root folder containing questions/ and scenes/")
    args = ap.parse_args()
    build_dataframe(args.split, args.clevr_root)


if __name__ == "__main__":
    main()
