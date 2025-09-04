#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step2_llama_7b_epoch1_lr9.65e-6
fi
mkdir -p $OUTPUT

deepspeed ../../main.py \
   --data_path shibing624/medical \
   --model_name_or_path /models/zxc/output_chris_medical_5 \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --lora_dim 32 \
   --lora_module_name "model.layers." \
   --output_dir $OUTPUT \
   --only_optimize_lora \
