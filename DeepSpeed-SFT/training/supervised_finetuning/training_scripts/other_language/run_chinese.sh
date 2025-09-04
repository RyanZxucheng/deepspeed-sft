#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi

mkdir -p $OUTPUT
# The Chinese data we found mostly only contain one response without another
# "rejected" response. Thus we only test the step 1 finetuning and use
# a data_split of 10,0,0 (keep all data for step 1).
# python -m deepspeed step1_supervised_finetuning.main \
deepspeed ../../main.py \
   --data_path krisfu/delicate_medical_r1_data_chinese \
   --model_name_or_path shenzhi-wang/Llama3.1-8B-Chinese-Chat \
   --gradient_accumulation_steps 8 \
   --lora_dim 4 \
   --only_optimize_lora \
   --print_loss \
   --zero_stage 3 \
   --deepspeed \
   --dtype bf16 \
   --output_dir $OUTPUT \
   --lora_module_name model.layers. \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --num_train_epochs 5 \
   --num_warmup_steps 20 \
   --max_seq_len 800 \

