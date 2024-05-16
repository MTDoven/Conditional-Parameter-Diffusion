accelerate launch --num_machines=1 --num_processes=1 --gpu_ids=7 \
  --main_process_port=36668 --dynamo_backend="no" --mixed_precision="no" \
  trainLoRA.py \
  --pretrained_model_name_or_path="../../datasets/PixArt-XL-256" \
  --dataset_name="../../datasets/OldStyles/style9" \
  --resolution=256 --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=100 --checkpointing_steps=10 \
  --learning_rate=5e-06 --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="lora_result_09_0" \
  --validation_prompt="Two women are rollerblading in front of some buses" --report_to="wandb" \
  --checkpoints_total_limit=5 --validation_epochs=5 \
  --rank=1
