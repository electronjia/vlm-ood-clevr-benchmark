# models to evaluate
MODELS = {
    "blip2": "Salesforce/blip2-opt-2.7b",
    "llava": "llava-hf/llava-v1.6-mistral-7b-hf",
    "qwen2vl": "Qwen/Qwen2.5-VL-7B-Instruct",
}

# paths
CLEVR_SCENES = "clevr/scenes/CLEVR_val_scenes.json"
CLEVR_IMAGES = "clevr/images/val/"
RESULTS_DIR  = "results/"

#  OOD split settings 
# Colors/shapes held out of the "seen" set → become OOD test data
HOLDOUT_COLORS = {"red", "gray"}
HOLDOUT_SHAPES = {"sphere"}
# MAX_SCENES = 2000       # cap per split so experiments don't take forever
SEED = 42

#  Inference settings 
MAX_NEW_TOKENS = 20
FEW_SHOT_K = 4          # number of in-context examples for few-shot
cfg_question_type = "query_shape"
