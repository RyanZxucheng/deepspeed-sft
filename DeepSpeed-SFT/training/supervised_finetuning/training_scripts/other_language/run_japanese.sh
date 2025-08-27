
deepspeed main.py \
   --data_path krisfu/delicate_medical_r1_data_chinese \
   --data_split 10,0,0 \
   --model_name_or_path shenzhi-wang/Llama3.1-8B-Chinese-Chat \
   --gradient_accumulation_steps 8 \
   --lora_dim 4 \
   --only_optimize_lora \
   --print_loss \
   --zero_stage 3 \
   --deepspeed \
   --dtype bf16 \
   --output_dir /models/zxc/output_chris_medical_10 \
   --lora_module_name model.layers. \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 8 \
   --num_train_epochs 10 \
   --num_warmup_steps 20 \
   --max_seq_len 800 \
