import os

#HF_HOME
#HF_DATASETS_CACHE



#salloc -N1 --gres=gpu:RTX_6000:1 -t0:45:00
#export HF_HOME=/home/hice1/PACEUSER/scratch/hfHOME
#export HF_DATASETS_CACHE=scratch/hugCache

#pip installed stuff after module load cuda


#Quadro RTX 6000
# Currently Loaded Modules:
#   1) pace-slurm/2022.06   4) mpc/1.2.1-zoh6w2  (H)   7) gcc/10.3.0-o57x6h             10) xz/5.2.2-kbeci4       (H)  13) slurm/current-4bdz7m  (H)
#   2) gmp/6.2.1-mw6xsf     5) zlib/1.2.7-s3gked (H)   8) libpciaccess/0.16-wfowrn (H)  11) libxml2/2.9.13-d4fgiv (H)  14) mvapich2/2.3.6-ouywal
#   3) mpfr/4.1.0-32gcbv    6) zstd/1.5.2-726gdz (H)   9) libiconv/1.16-pbdcxj     (H)  12) rdma-core/15-t43af2   (H)  15) cuda/12.1.1-6oacj6

#   Where:
#    H:  Hidden Module

print("starting")
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfifg, PeftModel
from trl import SFTTrainer


# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "llama-2-7b-miniguanaco"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 32
#lora rank


# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.3

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "/home/hice1/PACEUSER/scratch/pythonBazar"

# Number of training epochs
num_train_epochs = 2

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 1.4e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 25

# Log every X updates steps
logging_steps = 2

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = "auto"

# Load dataset (you can process it here)
#dataset = load_dataset(dataset_name, split="train")

#make the dataset 1/20th the size
#


#load dataset from parquet file 'custom.parquet'

dataset = load_dataset('parquet', data_files='/home/hice1/PACEUSER/scratch/conv-AI-project/generationAndQA.parquet', split='train')

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = True
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)


trainer.train()

import pandas as pd
a=pd.DataFrame(trainer.state.log_history)


#export the training info dataframe to csv
a.to_csv('/home/hice1/PACEUSER/scratch/conv-AI-project/modelNameInfo.csv')

#print the keys
print(a.keys())

#print the first few lines
print(a['train_loss'][:5])
#get
#visualize the training curves
#tensorboard --logdir=scratch/pythonBazar

# # Save tokenizer


# # Save trained model
# trainer.model.save_pretrained(new_model)

# #save trained modelf to path
trainer.model.save_pretrained("/home/hice1/PACEUSER/scratch/modelSave/modelName")


