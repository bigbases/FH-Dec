#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FreDec" ]; then
    mkdir ./logs/FreDec
fi

data_name=ETTm2
decomposer_name=FreDec

for pred_len in 12 24 48 96
do
for model_name in Fredformer
do
    # model_name에 former가 포함되어 있지 않으면 label_len=0으로 설정
    if [[ $model_name != *"former"* ]]; then
        label_len=0
    else
        label_len=48
    fi
    python -u run.py \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTm2.csv \
      --model_id FreDec_ETTm2_96_$pred_len \
      --model $model_name \
      --is_decomp 1 \
      --decomposer $decomposer_name \
      --kernel_size 25 \
      --freq_len 25 \
      --data $data_name \
      --d_model 24 \
      --d_ff 128 \
      --features MS \
      --seq_len 96 \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 29 \
      --dec_in 29 \
      --c_out 1 \
      --e_layers 2 \
      --patch_len 4 \
      --stride 4 \
      --des 'Exp' \
      --lradj 'type1'\
      --pct_start 0.4 \
      --cf_dim 128 \
      --cf_depth 2 \
      --cf_heads 8 \
      --cf_mlp 96 \
      --individual 0\
      --itr 1 --batch_size 32 --learning_rate 0.001 >logs/FreDec/$model_name'_'$data_name'_96_'$pred_len.log
done
done
