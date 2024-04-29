accelerate launch --num_machines=1 --num_processes=1 --gpu_ids=6 \
  --main_process_port=36730 --dynamo_backend="no" --mixed_precision="bf16" \
  trainLoRA.py \
  --pretrained_model_name_or_path="../../datasets/PixArt-XL-256" \
  --dataset_name="../../datasets/MultiStyles/style18" \
  --resolution=256 --random_flip \
  --train_batch_size=24 \
  --num_train_epochs=400 --checkpointing_steps=200 \
  --learning_rate=2e-05 --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --seed=2245 \
  --output_dir="lora_result_18_0" \
  --validation_prompt="Two women are rollerblading in front of some buses" --report_to="wandb" \
  --checkpoints_total_limit=50 --validation_epochs=100 \
  --rank=1 \
  #--resume_from_checkpoint="./lora_result_18_0/checkpoint-7400"