#!/bin/bash

nohup python -m realpdebench.train --config configs/cylinder/trainsolver.yaml --lambda_sg 0.01 --gpu 0 --train_data_type real --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > nohup_real.log 2>&1 &
nohup python -m realpdebench.train --config configs/cylinder/trainsolver.yaml --lambda_sg 0.01 --gpu 1 --train_data_type numerical --use_hf_dataset --hf_auto_download --hf_endpoint https://hf-mirror.com > nohup_numerical.log 2>&1 &