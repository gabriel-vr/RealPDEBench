#!/bin/bash

# FNO1d with semigroup regularization
nohup python -m realpdebench.eval --gpu 3 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_False/2026-07-09_15-26-19/model_4000.pth --use_hf_dataset > logs/eval/model_version_1/new_cylinder_data/nohup_eval_real_fno1d.log 2>&1 &
nohup python -m realpdebench.eval --gpu 4 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_numerical_False/2026-07-06_23-12-29/model_0080.pth --use_hf_dataset > logs/eval/model_version_1/new_cylinder_data/nohup_eval_numerical_fno1d.log 2>&1 &
nohup python -m realpdebench.eval --gpu 5 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_True/2026-07-07_15-17-14/model_0240.pth --use_hf_dataset > logs/eval/model_version_1/new_cylinder_data/nohup_eval_real_fno1d_finetune.log 2>&1 &

# FNO1d without semigroup regularization
nohup python -m realpdebench.eval --gpu 6 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_False/model_4000.pth --use_hf_dataset > logs/eval/model_version_1/new_cylinder_data/nohup_eval_real_fno1d_wo_sg.log 2>&1 &
nohup python -m realpdebench.eval --gpu 7 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_numerical_False/2026-07-07_16-38-19/model_0080.pth --use_hf_dataset > logs/eval/model_version_1/new_cylinder_data/nohup_eval_numerical_fno1d_wo_sg.log 2>&1 &
