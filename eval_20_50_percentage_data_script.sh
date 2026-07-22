#!/bin/bash

# FNO1d without semigroup regularization 50 percent data evaluation
nohup python -m realpdebench.eval --gpu 6 --config configs/cylinder/fno1d/fno1d_50_percent_data_eval.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_False/2026-07-13_17-06-37/model_4000.pth --use_hf_dataset > logs/eval/model_version_50_percent/nohup_eval_real_fno1d_wo_sg.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 7 --config configs/cylinder/fno1d/fno1d_50_percent_data_eval.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_numerical_False/2026-07-13_17-06-37/model_0100.pth --use_hf_dataset > logs/eval/model_version_50_percent/nohup_eval_numerical_fno1d_wo_sg.log 2>&1 &

# FNO1d without semigroup regularization 20 percent data evaluation
#nohup python -m realpdebench.eval --gpu 6 --config configs/cylinder/fno1d/fno1d_20_percent_data_eval.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_False/2026-07-13_17-08-21/model_4000.pth --use_hf_dataset > logs/eval/model_version_20_percent/nohup_eval_real_fno1d_wo_sg.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 7 --config configs/cylinder/fno1d/fno1d_20_percent_data_eval.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_numerical_False/2026-07-13_17-08-21/model_0100.pth --use_hf_dataset > logs/eval/model_version_20_percent/nohup_eval_numerical_fno1d_wo_sg.log 2>&1 &
