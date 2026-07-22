#!/bin/bash

# FNO1d with semigroup regularization
#nohup python -m realpdebench.eval --gpu 3 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_False/2026-07-12_12-54-24/model_4000.pth --use_hf_dataset > logs/eval/model_version_2/nohup_eval_real_fno1d.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 4 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_numerical_False/2026-07-12_11-32-27/model_0080.pth --use_hf_dataset > logs/eval/model_version_2/nohup_eval_numerical_fno1d.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 5 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_True/2026-07-15_18-33-19/model_1000.pth --use_hf_dataset > logs/eval/model_version_2/nohup_eval_real_fno1d_finetune_w_sg.log 2>&1 &

# FNO1d with semigroup regularization using real data for training
#nohup python -m realpdebench.eval --gpu 3 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_False/2026-07-15_18-34-37/model_4000.pth --use_hf_dataset > logs/eval/model_version_2/nohup_eval_real_fno1d_w_truth_data.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 1 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_numerical_False/2026-07-15_18-34-37/model_0160.pth --use_hf_dataset > logs/eval/model_version_2/nohup_eval_numerical_fno1d_w_truth_data.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 0 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_True/2026-07-16_23-54-10/model_1000.pth --use_hf_dataset > logs/eval/model_version_2/nohup_eval_real_fno1d_finetune_w_sg_truth_data.log 2>&1 &


# FNO1d without semigroup regularization
#nohup python -m realpdebench.eval --gpu 3 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_False/2026-07-11_23-32-25/model_4000.pth --use_hf_dataset > logs/eval/model_version_2/nohup_eval_real_fno1d_wo_sg.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 3 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_numerical_False/2026-07-11_23-32-25/model_0080.pth --use_hf_dataset > logs/eval/model_version_2/nohup_eval_numerical_fno1d_wo_sg.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 4 --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/fno1d/fno1d_cylinder_real_True/2026-07-13_10-02-31/model_1000.pth --use_hf_dataset > logs/eval/model_version_2/nohup_eval_real_fno1d_finetune_wo_sg.log 2>&1 &



# DeepONet with semigroup regularization
#nohup python -m realpdebench.eval --gpu 1 --config configs/cylinder/deeponet/deeponet.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/deeponet/deeponet_cylinder_real_False/2026-07-17_15-46-39/model_4800.pth --use_hf_dataset > logs/eval/deeponet/nohup_eval_real.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 2 --config configs/cylinder/deeponet/deeponet.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/deeponet/deeponet_cylinder_numerical_False/2026-07-17_15-46-38/model_0400.pth --use_hf_dataset > logs/eval/deeponet/nohup_eval_numerical.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 1 --config configs/cylinder/deeponet/deeponet.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/deeponet/deeponet_cylinder_real_True/2026-07-19_19-23-07/model_0920.pth --use_hf_dataset > logs/eval/deeponet/nohup_eval_real_finetune_w_sg.log 2>&1 &

# DeepONet with semigroup regularization using real data for training
#nohup python -m realpdebench.eval --gpu 3 --config configs/cylinder/deeponet/deeponet.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/deeponet/deeponet_cylinder_real_False/2026-07-17_16-05-33/model_5000.pth --use_hf_dataset > logs/eval/deeponet/nohup_eval_real_w_truth_data.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 4 --config configs/cylinder/deeponet/deeponet.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/deeponet/deeponet_cylinder_numerical_False/2026-07-17_16-05-34/model_0400.pth --use_hf_dataset > logs/eval/deeponet/nohup_eval_numerical_w_truth_data.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 0 --config configs/cylinder/deeponet/deeponet.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/deeponet/deeponet_cylinder_real_True/2026-07-19_19-23-58/model_0920.pth --use_hf_dataset > logs/eval/deeponet/nohup_eval_real_finetune_w_sg_truth_data.log 2>&1 &


# DeepONet without semigroup regularization
#nohup python -m realpdebench.eval --gpu 5 --config configs/cylinder/deeponet/deeponet.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/deeponet/deeponet_cylinder_real_False/2026-07-17_16-06-03/model_4600.pth --use_hf_dataset > logs/eval/deeponet/nohup_eval_real_wo_sg.log 2>&1 &
#nohup python -m realpdebench.eval --gpu 6 --config configs/cylinder/deeponet/deeponet.yaml --train_data_type numerical --checkpoint_path /raid/work/sandbox/RealPDEBench/models/deeponet/deeponet_cylinder_numerical_False/2026-07-17_16-06-03/model_0200.pth --use_hf_dataset > logs/eval/deeponet/nohup_eval_numerical_wo_sg.log 2>&1 &
nohup python -m realpdebench.eval --gpu 4 --config configs/cylinder/deeponet/deeponet.yaml --train_data_type real --checkpoint_path /raid/work/sandbox/RealPDEBench/models/deeponet/deeponet_cylinder_real_True/2026-07-19_19-24-28/model_0960.pth --use_hf_dataset > logs/eval/deeponet/nohup_eval_real_finetune_wo_sg.log 2>&1 &
