# Phi-3.5 Vision Fine-tuning on UCF101

Fine-tune Microsoft's Phi-3.5 Vision model for video action classification using LoRA on Azure ML.

## Dataset

**UCF101 (10-class subset)**: ApplyEyeMakeup, ApplyLipstick, Archery, BabyCrawling, BalanceBeam, BandMarching, BaseballPitch, Basketball, BasketballDunk, BenchPress

- 8 frames per video extracted as images
- Task: Classify action from temporal sequence

## Quick Start

### 1. Prepare Data
```bash
python convert_ucf101.py --out_dir ./converted_ucf101
```

Upload to Azure ML Datastore: `azureml://datastores/workspaceblobstore/paths/converted_ucf101/`

### 2. Configure Environment
Create `.env`:
```
SUBSCRIPTION_ID=your-subscription-id
RESOURCE_GROUP=your-resource-group
WORKSPACE_NAME=your-workspace-name
HF_TOKEN=your-huggingface-token
```

### 3. Run Training
Execute `phi-35-vision-ft-01.ipynb`:
- Build Docker environment
- Submit training job to Azure ML compute cluster

### 4. Deploy Model
- Register merged model from training outputs
- Create managed online endpoint
- Deploy with `score.py` for multi-frame inference

## Key Files

- `phi-35-vision-ft-01.ipynb`: Main training notebook
- `src/train.py`: LoRA fine-tuning script (TRL + PEFT)
- `src/phi3v_dataset.py`: UCF101 multi-frame dataset loader
- `src/score.py`: Inference endpoint handler
- `src/processor_patch.py`: Processor save resilience utilities

## Architecture

- **Base Model**: microsoft/phi-3.5-vision-instruct
- **Method**: LoRA
- **Target Modules**: Language (qkv_proj, o_proj, gate_up_proj) + Vision (q/k/v_proj, fc1/fc2, img_projection)
- **Precision**: BF16
- **Framework**: TRL SFTTrainer + Transformers

