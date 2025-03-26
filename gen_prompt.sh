#!/bin/bash

MODEL_PATH="./save/omnicontrol_ckpt/model_humanml3d.pt"
REPEATS=1
COND_MODE_TEXT="only_text"
COND_MODE_BOTH="both_text_spatial"
DENSITY=49  # 25% of 196 frames
TEXT_PROMPT="predefined"
TIMESTEP_RESPACING="ddim500"

echo "Generating ddim50"
python -m sample.generate \
    --model_path $MODEL_PATH \
    --text_prompt "$TEXT_PROMPT" \
    --timestep_respacing "ddim50" \
    --use_ddim True \
    --output_dir ./save/results/ddim50 \
    --num_repetitions $REPEATS

echo "Generating ddim100"
python -m sample.generate \
    --model_path $MODEL_PATH \
    --text_prompt "$TEXT_PROMPT" \
    --timestep_respacing "ddim100" \
    --use_ddim True \
    --output_dir ./save/results/ddim50 \
    --num_repetitions $REPEATS

echo "Generating ddim250"
python -m sample.generate \
    --model_path $MODEL_PATH \
    --text_prompt "$TEXT_PROMPT" \
    --timestep_respacing "ddim250" \
    --use_ddim True \
    --output_dir ./save/results/ddim250 \
    --num_repetitions $REPEATS

echo "Generating ddim500"
python -m sample.generate \
    --model_path $MODEL_PATH \
    --text_prompt "$TEXT_PROMPT" \
    --timestep_respacing "ddim500" \
    --use_ddim True \
    --output_dir ./save/results/ddim500 \
    --num_repetitions $REPEATS

echo "Generating dpm3_50steps"
python -m sample.generate \
    --model_path $MODEL_PATH \
    --text_prompt "$TEXT_PROMPT" \
    --dpm_steps 50 \
    --use_dpm True \
    --output_dir ./save/results/dpm3_50steps \
    --num_repetitions $REPEATS

echo "Generating dpm3_100steps"
python -m sample.generate \
    --model_path $MODEL_PATH \
    --text_prompt "$TEXT_PROMPT" \
    --dpm_steps 100 \
    --use_dpm True \
    --output_dir ./save/results/dpm3_100steps \
    --num_repetitions $REPEATS

echo "Generating dpm3_250steps"
python -m sample.generate \
    --model_path $MODEL_PATH \
    --text_prompt "$TEXT_PROMPT" \
    --dpm_steps 250 \
    --use_dpm True \
    --output_dir ./save/results/dpm3_250steps \
    --num_repetitions $REPEATS

echo "Generating dpm3_500steps"
python -m sample.generate \
    --model_path $MODEL_PATH \
    --text_prompt "$TEXT_PROMPT" \
    --dpm_steps 500 \
    --use_dpm True \
    --output_dir ./save/results/dpm3_500steps \
    --num_repetitions $REPEATS