#!/bin/sh
model_path=$1
# eval root joint
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 0 --density 1 --use_dpm True --dpm_order 2 --dpm_steps 75
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 0 --density 5 --use_dpm True --dpm_order 2 --dpm_steps 75
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 0 --density 25 --use_dpm True --dpm_order 2 --dpm_steps 75
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 0 --density 100 --use_dpm True --dpm_order 2 --dpm_steps 75

python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 0 --density 1 --use_dpm True --dpm_order 3 --dpm_steps 250
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 0 --density 1 --use_dpm True --dpm_order 3 --dpm_steps 100
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 0 --density 1 --use_dpm True --dpm_order 3 --dpm_steps 75
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 0 --density 1 --use_dpm True --dpm_order 3 --dpm_steps 500
python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 0 --density 1 --use_dpm True --dpm_order 3 --dpm_steps 50


# # eval left foot
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 10 --density 1
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 10 --density 2
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 10 --density 5
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 10 --density 25
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 10 --density 100
# # eval right foot
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 11 --density 1
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 11 --density 2
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 11 --density 5
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 11 --density 25
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 11 --density 100
# # eval head
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 1
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 2
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 5
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 25
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 15 --density 100
# # eval left wrist
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 1
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 2
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 5
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 25
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 20 --density 100
# # eval right wrist
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 1
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 2
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 5
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 25
# python -m eval.eval_humanml --model_path ${model_path} --eval_mode omnicontrol --control_joint 21 --density 100