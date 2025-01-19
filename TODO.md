./eval_omnicontrol_all.sh ./save/omnicontrol_ckpt/model_humanml3d.pt 
./eval_omnicontrol.sh ./save/omnicontrol_ckpt/model_humanml3d.pt 0 100
- length of motion variable?
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --text_prompt ''