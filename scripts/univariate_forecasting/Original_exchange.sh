#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


if [ ! -d "./logs/Original" ]; then
    mkdir ./logs/Original
fi

data_name=custom

for pred_len in 12 24 48
do
for model_name in TimeMixer iTransformer
do
    # model_name에 former가 포함되어 있지 않으면 label_len=0으로 설정
    if [[ $model_name != *"former"* ]]; then
        label_len=0
    else
        label_len=48
    fi
    python -u run.py \
      --is_training 1 \
      --root_path ./data/ \
      --data_path exchange.csv \
      --model_id Original_exchange_96_$pred_len \
      --model $model_name \
      --data $data_name \
      --d_model 24 \
      --d_ff 128 \
      --features S \
      --seq_len 96 \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --e_layers 2 \
      --patch_len 4 \
      --stride 4 \
      --des 'Exp' \
      --lradj 'type3'\
      --pct_start 0.4 \
      --cf_dim 128 \
      --cf_depth 2 \
      --cf_heads 8 \
      --cf_mlp 96 \
      --individual 0\
      --itr 1 --batch_size 32 --learning_rate 0.001 >logs/Original/$model_name'_'$data_name'_96_'$pred_len.log
done
done
