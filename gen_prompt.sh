#!/bin/bash

MODEL_PATH="./save/omnicontrol_ckpt/model_humanml3d.pt"
REPEATS=1
COND_MODE_TEXT="only_text"
COND_MODE_BOTH="text_and_spatial"
DENSITY=49  # 25% of 196 frames
TEXT_PROMPT="A person jumps forward and waves."
TIMESTEP_RESPACING="ddim250"

# === TEXT-ONLY MODE ===
echo "Running TEXT-ONLY without prompt decomposition..."
python -m sample.generate \
    --model_path $MODEL_PATH \
    --num_repetitions $REPEATS \
    --text_prompt "$TEXT_PROMPT" \
    --cond_mode $COND_MODE_TEXT \
    --timestep_respacing $TIMESTEP_RESPACING \
    --use_ddim True \
    --output_dir ./results/text_only_no_decomp

echo "Running TEXT-ONLY with prompt decomposition..."
python -m sample.generate \
    --model_path $MODEL_PATH \
    --num_repetitions $REPEATS \
    --text_prompt "$TEXT_PROMPT" \
    --cond_mode $COND_MODE_TEXT \
    --timestep_respacing $TIMESTEP_RESPACING \
    --use_prompt_decomposer True \
    --use_ddim True \
    --output_dir ./results/text_only_with_decomp


# === TEXT + SPATIAL MODE (25%) ===
echo "Running TEXT + SPATIAL (25%) without prompt decomposition..."
python -m sample.generate \
    --model_path $MODEL_PATH \
    --num_repetitions $REPEATS \
    --text_prompt "$TEXT_PROMPT" \
    --cond_mode $COND_MODE_BOTH \
    --keyframe_density $DENSITY \
    --timestep_respacing $TIMESTEP_RESPACING \
    --use_ddim True \
    --output_dir ./results/both_25_no_decomp

echo "Running TEXT + SPATIAL (25%) with prompt decomposition..."
python -m sample.generate \
    --model_path $MODEL_PATH \
    --num_repetitions $REPEATS \
    --text_prompt "$TEXT_PROMPT" \
    --cond_mode $COND_MODE_BOTH \
    --keyframe_density $DENSITY \
    --timestep_respacing $TIMESTEP_RESPACING \
    --use_prompt_decomposer True\
    --use_ddim True \
    --output_dir ./results/both_25_with_decomp
