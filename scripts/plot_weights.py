import torch
import matplotlib.pyplot as plt
import argparse

def extract_weight(path, arch_type="standard"):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    # --- Logic for standard Transformer (Teacher and Thief) ---
    if arch_type == "standard":
        target_key = 'layers.0.ffn.linear1.weight'
        if target_key in state_dict:
            return state_dict[target_key].detach().cpu().numpy().flatten()
    
    # --- Logic for Llama Architecture (Independent Scratch) ---
    elif arch_type == "llama":
        # HuggingFace Llama weights are usually model.layers.i.mlp.gate_proj.weight
        target_key = 'model.layers.0.mlp.gate_proj.weight'
        if target_key in state_dict:
            return state_dict[target_key].detach().cpu().numpy().flatten()
        
    # Fallback search if exact keys aren't found
    for key in state_dict.keys():
        if "weight" in key and ("ffn.linear1" in key or "mlp.gate_proj" in key):
            print(f"Found match: {key} in {path}")
            return state_dict[key].detach().cpu().numpy().flatten()
            
    raise KeyError(f"Could not find compatible weight layer in {path}")

def analyze_weights_triple(args):
    try:
        # 1. Extract weights
        t_weights = extract_weight(args.teacher_checkpoint, arch_type="standard")
        b_weights = extract_weight(args.student_checkpoint, arch_type="standard")
        c_weights = extract_weight(args.llama_checkpoint, arch_type="llama")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 2. Plotting
    plt.figure(figsize=(18, 6))

    # Teacher
    plt.subplot(1, 3, 1)
    plt.hist(t_weights, bins=100, color='blue', alpha=0.7, density=True)
    plt.title("Teacher (Spi-dLU)\nWeight Distribution")
    plt.xlim(-0.15, 0.15) # Keep scales consistent for visual proof

    # Thief
    plt.subplot(1, 3, 2)
    plt.hist(b_weights, bins=100, color='red', alpha=0.7, density=True)
    plt.title("Thief (Distilled GeLU)\nWeight Distribution")
    plt.xlim(-0.15, 0.15)

    # Llama
    plt.subplot(1, 3, 3)
    plt.hist(c_weights, bins=100, color='darkgreen', alpha=0.7, density=True)
    plt.title("Independent (Llama-Scratch)\nWeight Distribution")
    plt.xlim(-0.15, 0.15)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Final Weight DNA Analysis saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", default="models/teacher_spidlu/checkpoint_epoch_9.pt")
    parser.add_argument("--student_checkpoint", default="models/student_stolen_epoch_4.pt")
    parser.add_argument("--llama_checkpoint", default="models/llama_scratch/checkpoint_epoch_4.pt")
    parser.add_argument("--output", default="experiments/weight_dna_with_llama.png")
    analyze_weights_triple(parser.parse_args())