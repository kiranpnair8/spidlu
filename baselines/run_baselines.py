import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from spikingjelly.activation_based import functional
from modules.spidlu_layer import SpiDLU
from utils.data_utils import get_dataloader, log_metrics

# --- SPIKING SURGERY WRAPPERS ---

class SpiLlamaMLP(nn.Module):
    """Replacement for Llama/Qwen SwiGLU MLP"""
    def __init__(self, original_mlp, alpha, threshold, T):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.spidlu = SpiDLU(alpha=alpha, threshold=threshold, T=T)

    def forward(self, x):
        # Llama Gating: SpiDLU replaces SiLU in the gated path
        # Note: We apply SpiDLU to the product to create the temporal manifold
        gated = self.gate_proj(x) * self.up_proj(x)
        return self.down_proj(self.spidlu(gated))

class SpiPhiMLP(nn.Module):
    """Specific wrapper for Phi-1.5 which uses fc1/fc2 instead of gate/up/down"""
    def __init__(self, original_mlp, alpha, threshold, T):
        super().__init__()
        self.fc1 = original_mlp.fc1
        self.fc2 = original_mlp.fc2
        self.spidlu = SpiDLU(alpha=alpha, threshold=threshold, T=T)

    def forward(self, x):
        # Phi-1.5 math: fc2(activation(fc1(x)))
        return self.fc2(self.spidlu(self.fc1(x)))

def inject_spidlu(model, model_type, alpha, threshold, T):
    print(f"--- Performing Surgery on {model_type} ---")
    
    # Get the layers regardless of model name
    base_model = getattr(model, model.base_model_prefix)
    layers = base_model.layers if hasattr(base_model, 'layers') else base_model.h
    
    for i in range(len(layers)):
        original_mlp = layers[i].mlp
        
        # Check if it's a SwiGLU model (Llama/Qwen) or a Linear model (Phi)
        if hasattr(original_mlp, 'gate_proj'):
            layers[i].mlp = SpiLlamaMLP(original_mlp, alpha, threshold, T)
        elif hasattr(original_mlp, 'fc1'):
            layers[i].mlp = SpiPhiMLP(original_mlp, alpha, threshold, T)
        else:
            raise AttributeError(f"Unknown MLP structure in {model_type}")
            
    return model

# --- MAIN RUNNER ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--model_name", type=str, choices=["llama", "phi", "qwen"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Pretrained Baseline
    print(f"Loading {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager" 
    )

    model.gradient_checkpointing_enable()
    torch.cuda.empty_cache()
    

    # 2. Inject SpiDLU (The Surgery)
    # Hardcoded spidlu params from your config
    ALPHA, THRESHOLD, T_STEPS = 0.9, 1.0, 4
    model = inject_spidlu(model, args.model_name, ALPHA, THRESHOLD, T_STEPS)
    model.to(device)

    # 3. Data Loading
    config = AutoConfig.from_pretrained(args.model_path)
    max_len = getattr(config, "max_position_embeddings", 1024)
    print(f"Model Max Position Embeddings: {max_len}")

    # Pass this max_len to your get_dataloader
    # Change this line in run_baselines.py
    train_loader, val_loader, _ = get_dataloader(
        {
            'data': {'dataset_name': 'wikitext', 'dataset_config': 'wikitext-2-v1'},
            'model': {'max_seq_len': min(max_len, 512)}, 
            'training': {'batch_size': args.batch_size}
        },
        args.model_path  
    )

    # 4. Optimizer (Alignment Tuning)
    if "qwen" in args.model_name.lower():
        print("--- Using SGD Optimizer for Qwen memory efficiency ---")
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            momentum=0.9
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr
        )
        
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training Loop
    model.train()
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)

            optimizer.zero_grad()
            functional.reset_net(model) 
            
            outputs = model(inputs).logits
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            if "qwen" in args.model_name.lower():
                torch.cuda.empty_cache()
            optimizer.step()

            # Logging
            if i % 10 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")

            # --- NEW: INTERMEDIATE VALIDATION & FREQUENT SAVING ---
            # Reduced interval to 400 to ensure we catch saves on small datasets
            if i > 0 and i % 400 == 0:
                model.eval()
                val_loss = 0
                print(f"--- Running Validation at Batch {i} ---")
                with torch.no_grad():
                    for v_idx, v_batch in enumerate(val_loader):
                        if v_idx > 100: break 
                        functional.reset_net(model)
                        v_out = model(v_batch['input_ids'].to(device)).logits
                        val_loss += criterion(v_out.view(-1, v_out.size(-1)), v_batch['labels'].to(device).view(-1)).item()
                
                avg_val_loss = val_loss / 100
                ppl = torch.exp(torch.tensor(avg_val_loss))
                print(f"--- Batch {i} | Val PPL: {ppl:.2f} ---")
                
                # Intermediate Checkpoint
                save_path = f"models/alignment_{args.model_name}_epoch{epoch}_batch{i}"
                model.save_pretrained(save_path)
                print(f"Checkpoint saved to {save_path}")
                
                model.train()

        # --- CRITICAL: SAVE AT THE END OF EVERY EPOCH ---
        # This catches the model even if the total batches are less than 2000
        epoch_save_path = f"models/alignment_{args.model_name}_epoch_{epoch}_final"
        model.save_pretrained(epoch_save_path)
        print(f"End of Epoch {epoch} - Model saved to {epoch_save_path}")

    # --- FINAL SAFETY SAVE ---
    final_path = f"models/{args.model_name}_aligned_complete"
    model.save_pretrained(final_path)
    print(f"Full alignment complete. Final model saved to {final_path}")

if __name__ == "__main__":
    main()