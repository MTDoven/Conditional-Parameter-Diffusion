# Codes of vision parts in [Conditional LoRA Parameter Generation](https://arxiv.org/abs/2408.01415).

## Usage

### 1. Prepare your LoRA parameter Dataset.
```bash
export WANDB_API_KEY="Set your wandb key here"
cd PixArt-StyleTrans-Comp
```
Modify trainLoRA.sh  
You need to confirm the following arguments:
- pretrained_model_name_or_path: path to your [PixArt-XL-256](https://github.com/PixArt-alpha/PixArt-alpha) model.
- dataset_name: a folder filled with images and the name of the image files are prompts.
- output_dir: a path to save the output of LoRA parameters. It follows a rule lora_result_{style_class}_{param_group}.
```bash
sh trainLoRA.sh
```
Modify constructLoRA.py  
You need to confirm the following arguments:
- style_class: the style_class in lora_result_{style_class}_{param_group}.
- param_group: the param_group in lora_result_{style_class}_{param_group}.
```bash
python constructLoRA.py
```
Now, you should have got a LoRA parameter dataset folder: `CheckpointTrainLoRA`.


### 2. Train CondPDiff
Train AE.  
Modify trainVAE-Transfer.py  
You need to confirm the following arguments:  
- image_data_path: path to your image dataset which is used as condition.
- lora_data_path: path to your `CheckpointTrainLoRA`.
- result_save_path: path to save your AE checkpoint.
```bash
cd ../CondiPDiff
python trainVAE-Transfer.py
python evaluateVAE-Transfer.py
```
Train Latent Diffusion.  
Modify trainDDPM-Transfer.py  
You need to confirm the following arguments:  
- image_data_path: path to your image dataset which is used as condition.
- lora_data_path: path to your `CheckpointTrainLoRA`.
- vae_checkpoint_path: path to your AE checkpoint.
- result_save_path: path to save your Diffusion checkpoint.
```bash
python trainDDPM-Transfer.py
python evaluateDDPM-Transfer.py
```

### Note
The tag: `Transfer` refers to `StyleTrans-Comp`, which verified the conditional generation ability.  
The tag: `Continue` refers to `StyleTrans-Conti`, which verified the in-set generalization ability.  
The tag: `Classify` refers to `Classify-CIFAR10`, which is an experiment for conditional generation by image class.  
The example above showed the usage of tag `Transfer`, the others follow a similar method to use.



