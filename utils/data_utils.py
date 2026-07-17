import torch
import csv
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_dataloader(cfg, model_path): # Pass model_path here
    """
    Loads, tokenizes, and groups text with strict truncation to prevent CUDA Assert errors.
    """
    # 1. Use the ACTUAL model tokenizer (Critical!)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Dataset
    dataset = load_dataset(cfg['data']['dataset_name'], cfg['data']['dataset_config'])
    
    # 3. Block size from config
    block_size = cfg['model']['max_seq_len']

    # 4. Tokenize with strict truncation
    def tokenize_function(examples):
        # We truncate here as a first line of defense
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=block_size,
            return_overflowing_tokens=False # Change to True if you want to keep all fragments
        )

    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names, # Remove all original columns
        num_proc=4
    )

    # 5. Group text into fixed-length blocks
    def group_texts(examples):
        # Concatenate all fragments
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Ensure total_length is a multiple of block_size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        # Causal LM labels are exactly the input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4
    )

    # 6. Set format for PyTorch
    lm_datasets.set_format("torch")

    train_loader = DataLoader(
        lm_datasets["train"], 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True
    )
    
    val_loader = DataLoader(
        lm_datasets["validation"], 
        batch_size=cfg['training']['batch_size']
    )

    return train_loader, val_loader, tokenizer

def log_metrics(epoch, train_loss, val_loss, val_ppl, **kwargs):
    # This handles both 'filepath' (from your function) and 'filename' (from the script)
    filepath = kwargs.get('filename', kwargs.get('filepath', "experiments/teacher_training_log.csv"))
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Check if file exists to write header
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_ppl'])  # Header
        writer.writerow([epoch, train_loss, val_loss, val_ppl])