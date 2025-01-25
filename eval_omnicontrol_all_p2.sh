#!/bin/sh
model_path=$1
# eval head
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 1
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 2
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 5
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 25
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 100
# eval left wrist
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 1
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 2
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 5
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 25
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 100
# eval right wrist
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 1
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 2
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 5
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 25
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 100