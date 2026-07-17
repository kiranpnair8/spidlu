import os
import yaml
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from utils.data_utils import get_dataloader, log_metrics
from modules.transformer_block import SpiDLU_Transformer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate(model, val_loader, criterion, device):
    """Perform evaluation and calculate Perplexity."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            
            # Crucial: Reset neuronal states for each new sequence
            functional.reset_net(model)
            
            output = model(inputs)
            
            # CrossEntropy expects [Batch * Seq, Vocab]
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity

def train():
    # 1. Setup Environment
    cfg = load_config('configs/transformer_config.yaml')
    train_loader, val_loader, tokenizer = get_dataloader(cfg)
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else "cpu")
    
    # Ensure save directory exists
    save_dir = "models/teacher_spidlu"
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Initialize Spi-DLU Transformer
    model = SpiDLU_Transformer(
        vocab_size=len(tokenizer),
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead'],
        num_layers=cfg['model']['num_layers'],
        d_ff=cfg['model']['d_ff'],
        alpha=cfg['spidlu']['alpha'],
        threshold=cfg['spidlu']['threshold'],
        T=cfg['spidlu']['T_steps']
    ).to(device)

    print(f"--- Spi-DLU Teacher Training Started ---")
    print(f"Dynamics: T={cfg['spidlu']['T_steps']} steps, Threshold={cfg['spidlu']['threshold']}")

    # 3. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(cfg['training']['learning_rate']),
        weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    for epoch in range(cfg['training']['epochs']):
        model.train()
        epoch_loss = 0
        
        for i, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)

            optimizer.zero_grad()

            # reset_net is vital for Spiking Neural Networks to clear membrane potentials
            functional.reset_net(model)
            
            # Forward pass: Integrating information over T-steps
            output = model(inputs)
            
            # Flatten outputs for Language Modeling loss
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            
            # Surrogate Gradient Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch [{epoch}/{cfg['training']['epochs']}] | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # --- Post-Epoch Evaluation ---
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_ppl = validate(model, val_loader, criterion, device)
        
        print(f"\n>> Epoch {epoch} Summary:")
        print(f"   Avg Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:       {val_loss:.4f}")
        print(f"   Perplexity:     {val_ppl:.2f}\n")

        # 5. Logging and Checkpointing
        log_metrics(epoch, avg_train_loss, val_loss, val_ppl.item())
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'perplexity': val_ppl.item()
        }, checkpoint_path)
        
        print(f"Saved Checkpoint: {checkpoint_path}\n" + "-"*40)

if __name__ == "__main__":
    train()