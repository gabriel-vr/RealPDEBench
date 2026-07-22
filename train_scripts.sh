#!/bin/bash

# Trainsolver with semigroup regularization
#nohup python -m realpdebench.train --config configs/cylinder/trainsolver.yaml --lambda_sg 0.01 --gpu 1 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > nohup_numerical.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/trainsolver.yaml --lambda_sg 0.01 --gpu 0 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > nohup_real.log 2>&1 &

# FNO1d with semigroup regularization
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --lambda_sg 0.1 --gpu 5 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_2/nohup_numerical_fno1d.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --lambda_sg 0.1 --gpu 6 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_2/nohup_real_fno1d.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_w_sg_finetune.yaml --is_finetune --lambda_sg 0.1 --gpu 6 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_2/nohup_real_fno1d_finetune.log 2>&1 &

# FNO1d with semigroup regularization using ground truth data for semigroup regularization
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --use_ground_truth_for_sg --lambda_sg 0.1 --gpu 4 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_2/nohup_numerical_fno1d_use_ground_truth.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --use_ground_truth_for_sg --lambda_sg 0.1 --gpu 5 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_2/nohup_real_fno1d_use_ground_truth.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_w_sg_w_ground_truth_finetune.yaml --is_finetune --lambda_sg 0.1 --gpu 4 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_2/nohup_real_fno1d_w_sg_w_ground_truth_finetune.log 2>&1 &


# FNO1d without semigroup regularization
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --lambda_sg 0 --gpu 2 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_2/nohup_numerical_fno1d_wo_sg.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_long_evaluation.yaml --lambda_sg 0 --gpu 3 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_2/nohup_real_fno1d_wo_sg.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_wo_sg_finetune.yaml --is_finetune --lambda_sg 0 --gpu 4 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_2/nohup_real_fno1d_finetune_wo_sg.log 2>&1 &



# DeepONet with semigroup regularization
#nohup python -m realpdebench.train --config configs/cylinder/deeponet/deeponet.yaml --lambda_sg 0.1 --gpu 0 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/deeponet/nohup_numerical_deeponet.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/deeponet/deeponet.yaml --lambda_sg 0.1 --gpu 1 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/deeponet/nohup_real_deeponet.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/deeponet/deeponet_w_sg_finetune.yaml --is_finetune --lambda_sg 0.1 --gpu 6 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/deeponet/nohup_real_deeponet_finetune.log 2>&1 &

# DeepONet with semigroup regularization using ground truth data for semigroup regularization
#nohup python -m realpdebench.train --config configs/cylinder/deeponet/deeponet.yaml --use_ground_truth_for_sg --lambda_sg 0.1 --gpu 2 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/deeponet/nohup_numerical_fno1d_use_ground_truth.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/deeponet/deeponet.yaml --use_ground_truth_for_sg --lambda_sg 0.1 --gpu 3 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/deeponet/nohup_real_fno1d_use_ground_truth.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/deeponet/deeponet_w_sg_w_ground_truth_finetune.yaml --is_finetune --lambda_sg 0.1 --gpu 7 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/deeponet/nohup_real_w_sg_w_ground_truth_finetune.log 2>&1 &


# FNO1d without semigroup regularization
#nohup python -m realpdebench.train --config configs/cylinder/deeponet/deeponet.yaml --lambda_sg 0 --gpu 5 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/deeponet/nohup_numerical_fno1d_wo_sg.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/deeponet/deeponet.yaml --lambda_sg 0 --gpu 7 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/deeponet/nohup_real_fno1d_wo_sg.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/deeponet/deeponet_wo_sg_finetune.yaml --is_finetune --lambda_sg 0 --gpu 5 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/deeponet/nohup_real_finetune_wo_sg.log 2>&1 &
