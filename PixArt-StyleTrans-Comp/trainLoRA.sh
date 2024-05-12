accelerate launch --num_machines=1 --num_processes=1 --gpu_ids=5 \
  --main_process_port=36679 --dynamo_backend="no" --mixed_precision="no" \
  trainLoRA.py \
  --pretrained_model_name_or_path="../../datasets/PixArt-XL-256" \
  --dataset_name="../../datasets/FIDStyles/style3" \
  --resolution=256 --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=8 --checkpointing_steps=5 \
  --learning_rate=2e-05 --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --seed=41762 \
  --output_dir="lora_result_3_0" \
  --validation_prompt="Two women are rollerblading in front of some buses" --report_to="wandb" \
  --checkpoints_total_limit=64 --validation_epochs=50 \
  --rank=1
