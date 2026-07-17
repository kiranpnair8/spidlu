import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from modules.transformer_block import SpiDLU_Transformer
from utils.data_utils import get_dataloader, log_metrics
from spikingjelly.activation_based import functional

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    Hybrid Loss: KL Divergence (Teacher-Student) + CrossEntropy (Student-GroundTruth)
    """
    # Soft loss (KL Divergence)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T ** 2)
    
    # Hard loss (Standard Cross Entropy)
    hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    
    return alpha * soft_loss + (1 - alpha) * hard_loss

def train_student():
    # 1. Setup
    cfg = load_config('configs/transformer_config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, tokenizer = get_dataloader(cfg)
    
    # 2. Initialize Teacher (Spi-dLU)   
    teacher = SpiDLU_Transformer(
        vocab_size=len(tokenizer),
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead'],
        num_layers=cfg['model']['num_layers'],
        d_ff=cfg['model']['d_ff'],
        alpha=cfg['spidlu']['alpha'],
        threshold=cfg['spidlu']['threshold'],
        T=cfg['spidlu']['T_steps']
    ).to(device)
    
    checkpoint = torch.load("models/teacher_spidlu_watermarked/branded_checkpoint_epoch_1.pt")
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval() # Teacher is frozen
    print("Locked Spi-dLU Teacher loaded successfully.")

    # 3. Initialize Student (Standard GeLU - No Spiking)    
    # use a separate class that uses GeLU instead of SpiDLU layers.
    student = SpiDLU_Transformer(
        vocab_size=len(tokenizer),
        d_model=cfg['model']['d_model'],
        nhead=cfg['model']['nhead'],
        num_layers=cfg['model']['num_layers'],
        d_ff=cfg['model']['d_ff'],
        alpha=0, 
        threshold=0,
        T=1,     # Student usually has T=1 (non-spiking)
        use_spiking=False 
    ).to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    
    print(f"Starting Attack: Distilling into GeLU Student...")

    os.makedirs("models/student_branded", exist_ok=True)

    # 4. Distillation Loop
    for epoch in range(5): # Distillation usually needs fewer epochs
        student.train()
        total_loss = 0
        
        for i, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward Teacher 
            with torch.no_grad():
                functional.reset_net(teacher)
                teacher_logits = teacher(inputs)
            
            # Forward Student
            student_logits = student(inputs)
            
            # Calculate Distillation Loss
            loss = distillation_loss(student_logits, teacher_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} | Batch {i} | Distill Loss: {loss.item():.4f}")

        # Save the Model
        torch.save(student.state_dict(), f"models/student_branded/student_stolen_epoch_{epoch}.pt")

    # --- End of Epoch Validation ---
        student.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                v_inputs = batch['input_ids'].to(device)
                v_labels = batch['labels'].to(device)
                v_logits = student(v_inputs)
                v_loss = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), v_labels.view(-1))
                val_loss += v_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"--- Epoch {epoch} Student Val Loss: {avg_val_loss:.4f} | PPL: {torch.exp(torch.tensor(avg_val_loss)):.2f} ---")

if __name__ == "__main__":
    train_student()