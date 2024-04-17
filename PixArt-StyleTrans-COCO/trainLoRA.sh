accelerate launch --num_machines=1 --num_processes=1 --gpu_ids=2 \
  --main_process_port=36699 --dynamo_backend="no" --mixed_precision="bf16" \
  trainLoRA.py \
  --pretrained_model_name_or_path="./PixArt-XL-256" \
  --dataset_name="../../datasets/Styles/style4" \
  --resolution=256 --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=450 --checkpointing_steps=5 \
  --learning_rate=8e-06 --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --seed=41749 \
  --output_dir="lora_result_4_0" \
  --validation_prompt="Two women are rollerblading in front of some buses" --report_to="wandb" \
  --checkpoints_total_limit=128 --validation_epochs=90 \
  --rank=1
