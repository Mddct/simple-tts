## WIP

## Training

### 0 Data Prepare

```bash
# train tts llm
{"wav": "/data/BAC009S0764W0121.wav", "txt": "甚至出现交易几乎停滞的情况"}
{"wav": "/data/BAC009S0764W0122.wav", "txt": "一二线城市虽然也处于调整中"}
```
```bash
# train tts DIT
{"wav": "/data/BAC009S0764W0121.wav"}
{"wav": "/data/BAC009S0764W0122.wav"}
```


###  1 (Optional) train ssl  ctc vq (not support yet)
TODO
- [x] https://github.com/xingchensong/S3Tokenizer
- [ ] BestRQ + ctc + vq (future)

### 2 train llm on speech tokens with  text and spk condition
- [ ] tokenizer: char + bpe
- [x] generate
``` bash
output_dir=s1_output
# campplus.onnx path
spk_emb_onnx=....
torchrun --standalone --nnodes=1 --nproc_per_node=8 simpletts/train_llm.py \
    --data_path train.jsonl \
    --eval_data_path eval.jsonl \
    --bf16 True \
    --output_dir $output_dir \
    --max_steps 100000 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --save_steps 8000 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_steps 25000 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 10 \
    --campplus_onnx_path $spk_emb_onnx \
    --logging_steps=500 \
    --deepspeed ds_config_zero1.json
```

###  2 (Optional) train streaming flow matching or DIT (one-step)
- [x] rectified flow training
- [ ] rectified flow generate (**Ongoing**)

``` bash
output_dir=s2_output
# campplus.onnx path
spk_emb_onnx=....
torchrun --standalone --nnodes=1 --nproc_per_node=8 simpletts/train_dit.py \
    --data_path train.jsonl \
    --eval_data_path eval.jsonl \
    --bf16 True \
    --output_dir $output_dir \
    --max_steps 100000 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --save_steps 8000 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_steps 25000 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 10 \
    --campplus_onnx_path $spk_emb_onnx \
    --logging_steps=500 \
    --deepspeed ds_config_zero1.json
```



### 3 (Optional) train low latency streaming HIFIFAN or Vocos
TODO

## Inference
- [ ] vllm
- [ ] sglang
