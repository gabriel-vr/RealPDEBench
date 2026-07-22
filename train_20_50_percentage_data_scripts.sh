#!/bin/bash

# FNO1d without semigroup regularization 50 percent data
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_50_percent_data.yaml --lambda_sg 0 --gpu 2 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_50_percent/nohup_numerical_fno1d_wo_sg_50_percent.log 2>&1 &
#nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_50_percent_data.yaml --lambda_sg 0 --gpu 3 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_50_percent/nohup_real_fno1d_wo_sg_50_percent.log 2>&1 &


# FNO1d without semigroup regularization 20 percent data
nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_20_percent_data.yaml --lambda_sg 0 --gpu 2 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_20_percent/nohup_numerical_fno1d_wo_sg_20_percent.log 2>&1 &
nohup python -m realpdebench.train --config configs/cylinder/fno1d/fno1d_20_percent_data.yaml --lambda_sg 0 --gpu 3 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > logs/train/model_version_20_percent/nohup_real_fno1d_wo_sg_20_percent.log 2>&1 &
