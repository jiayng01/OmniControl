#!/bin/sh
model_path=$1
joint=$2
density=$3
use_ddim=$4
timestep_respacing=$5
# eval root joint
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint ${joint} --density ${density} --use_ddim ${use_ddim} --timestep_respacing ${timestep_respacing}
