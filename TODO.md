./eval_omnicontrol_all.sh ./save/omnicontrol_ckpt/model_humanml3d.pt 
./eval_omnicontrol.sh ./save/omnicontrol_ckpt/model_humanml3d.pt 0 100
./eval_omnicontrol.sh ./save/omnicontrol_ckpt/model_humanml3d.pt 0 100 --use_ddim True --timestep_respacing ddim500
- length of motion variable?
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --text_prompt ''
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --text_prompt '' --use_dpm True
python -m train.train_mdm --save_dir save/my_omnicontrol --dataset humanml --num_steps 400000 --batch_size 64 --resume_checkpoint ./save/model000475000.pt --lr 1e-5

- convert the text into prompts broken down

- limitations: prompt (if improves) does not cover spatial_only case
- max_text_len = 20
- spatial guidance after is better for dpm