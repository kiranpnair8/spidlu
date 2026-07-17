import torch
import torch.nn as nn
import yaml
import os
import argparse
from modules.transformer_block import SpiDLU_Transformer
from utils.data_utils import get_dataloader, set_seed
from transformers import LlamaConfig, LlamaForCausalLM

def train_baseline(arch_type="standard", config_path='configs/transformer_config.yaml', save_dir=None):
    # 1. Setup
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg['training'].get('seed'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, tokenizer = get_dataloader(cfg)
    vocab_size = len(tokenizer)
    
    # 2. Initialize Model C with Toggle
    if arch_type == "llama":
        print("Initializing Model C: Llama-1.1B Architecture (Scratch)...")
        # Configure a 1.1B Llama-style model
        llama_config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=512,           # 1.1B scale standard
            intermediate_size=1376,     # SwiGLU ratio
            num_hidden_layers=6,       # Depth
            num_attention_heads=8,     # Width
            max_position_embeddings=512, # Match your context window
            rms_norm_eps=1e-5
        )
        model_c = LlamaForCausalLM(llama_config).to(device)
        save_dir = save_dir or cfg['training'].get('llama_save_dir', "models/llama_scratch")
    else:
        print("Initializing Model C: Standard Transformer (Scratch)...")
        model_c = SpiDLU_Transformer(
            vocab_size=vocab_size,
            d_model=cfg['model']['d_model'],
            nhead=cfg['model']['nhead'],
            num_layers=cfg['model']['num_layers'],
            d_ff=cfg['model']['d_ff'],
            use_spiking=False 
        ).to(device)
        save_dir = save_dir or cfg['training'].get('gelu_save_dir', "models/gelu_scratch")
    
    optimizer = torch.optim.AdamW(
        model_c.parameters(),
        lr=float(cfg['training'].get('baseline_learning_rate', 5e-5))
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Training for Model C ({arch_type})...")

    # 3. Training Loop
    for epoch in range(cfg['training'].get('baseline_epochs', 5)):
        model_c.train()
        for i, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # HuggingFace Llama returns a CausalLMOutputWithPast object
            if arch_type == "llama":
                outputs = model_c(inputs).logits
            else:
                outputs = model_c(inputs)
                
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"[{arch_type}] Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")

        # Save Baseline
        os.makedirs(save_dir, exist_ok=True)
        torch.save({'model_state_dict': model_c.state_dict()}, 
                   f"{save_dir}/checkpoint_epoch_{epoch}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer_config.yaml")
    parser.add_argument("--arch", type=str, default="standard", choices=["standard", "llama"],
                        help="Toggle between standard transformer and Llama-1.1B")
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    
    train_baseline(arch_type=args.arch, config_path=args.config, save_dir=args.save_dir)