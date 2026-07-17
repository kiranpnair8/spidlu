import torch
import torch.nn.functional as F
import yaml
import os
import argparse
from modules.transformer_block import SpiDLU_Transformer
from utils.data_utils import get_dataloader, set_seed
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

def train_student(config_path='configs/transformer_config.yaml', teacher_checkpoint=None, save_dir=None):
    # 1. Setup
    cfg = load_config(config_path)
    set_seed(cfg['training'].get('seed'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, tokenizer = get_dataloader(cfg)
    teacher_checkpoint = teacher_checkpoint or cfg['training'].get(
        'teacher_checkpoint',
        "models/teacher_spidlu_watermarked/branded_checkpoint_epoch_1.pt"
    )
    save_dir = save_dir or cfg['training'].get('student_save_dir', "models/student_branded")
    
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
    
    checkpoint = torch.load(teacher_checkpoint, map_location=device)
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
    
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg['training'].get('student_learning_rate', 1e-4))
    )
    
    print(f"Starting Attack: Distilling into GeLU Student...")

    os.makedirs(save_dir, exist_ok=True)

    # 4. Distillation Loop
    for epoch in range(cfg['training'].get('student_epochs', 5)): # Distillation usually needs fewer epochs
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
        torch.save(student.state_dict(), os.path.join(save_dir, f"student_stolen_epoch_{epoch}.pt"))

    # --- End of Epoch Validation ---
        student.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                v_inputs = batch['input_ids'].to(device)
                v_labels = batch['labels'].to(device)
                v_logits = student(v_inputs)
                v_loss = F.cross_entropy(
                    v_logits.view(-1, v_logits.size(-1)),
                    v_labels.view(-1),
                    reduction='sum'
                )
                val_loss += v_loss.item()
                val_tokens += v_labels.numel()
        
        avg_val_loss = val_loss / max(val_tokens, 1)
        print(f"--- Epoch {epoch} Student Val Loss: {avg_val_loss:.4f} | PPL: {torch.exp(torch.tensor(avg_val_loss)):.2f} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer_config.yaml")
    parser.add_argument("--teacher_checkpoint", default=None)
    parser.add_argument("--save_dir", default=None)
    args = parser.parse_args()
    train_student(
        config_path=args.config,
        teacher_checkpoint=args.teacher_checkpoint,
        save_dir=args.save_dir
    )