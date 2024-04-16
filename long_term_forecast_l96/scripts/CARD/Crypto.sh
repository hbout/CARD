if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi


export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY=193225791e14193e33ee13219d82c506bca9005a
export WANDB_MODE=online

model_name=CARD
pred_lens=(1 2 4 8)
cuda_ids1=(0 0 0 0)
columns=470

for ((i = 0; i < 4; i++)) 
do 
    pred_len=${pred_lens[i]}
    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --factor 3 \
    --enc_in $columns \
    --dec_in $columns \
    --c_out $columns \
    --des 'Exp' \
    --itr 1 \
    --e_layers 2 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --patience 20\
    --train_epochs 100 --lradj CARD \
    --itr 1 --batch_size 128 --learning_rate 0.0001 \
    --dp_rank 8 --top_k 5   --mask_rate 0 --warmup_epochs 0 \
    2>&1 | tee logs/LongForecasting/$model_name'_'Crypto_96_$pred_len'.log' & \

done

# enc_in 入力チャネル数、 c_out 出力チャネル数
