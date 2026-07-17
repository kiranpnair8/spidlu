import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from modules.transformer_block import SpiDLU_Transformer
from transformers import LlamaConfig, LlamaForCausalLM
from spikingjelly.activation_based import functional
from utils.data_utils import set_seed

def get_neuron_trace(model, input_ids, is_spiking=False, is_llama=False):
    trace = []
    def hook_fn(module, input, output):
        data = output[0] if isinstance(output, tuple) else output
        # Average across all neurons in the layer to see the "Layer Pulse"
        trace.append(data.detach().cpu().numpy().mean(axis=-1))

    if is_llama:
        target = model.model.layers[0].mlp.act_fn
    else:
        target = model.layers[0].ffn.activation
    
    hook = target.register_forward_hook(hook_fn)
    with torch.no_grad():
        if is_spiking: functional.reset_net(model)
        _ = model(input_ids)
    hook.remove()
    return np.array(trace[0]).flatten()

def run_ghost_detector(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = args.vocab_size
    seq_len = args.seq_len
    
    # 1. Load Models
    teacher = SpiDLU_Transformer(vocab_size=vocab_size, use_spiking=True).to(device)
    teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device)['model_state_dict'])
    
    thief = SpiDLU_Transformer(vocab_size=vocab_size, use_spiking=False).to(device)
    thief.load_state_dict(torch.load(args.student_checkpoint, map_location=device))
    
    
    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1376,  # This matches your trained checkpoint
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=512
    )
    llama = LlamaForCausalLM(llama_config).to(device)
    
    # Load the checkpoint
    ckpt_c = torch.load(args.llama_checkpoint, map_location=device)
    llama.load_state_dict(ckpt_c['model_state_dict'])
    llama.eval()

    # 2. Pick a fixed random prompt for reproducibility
    prompt = torch.randint(0, vocab_size, (1, seq_len)).to(device)

    # 3. Get Temporal Traces
    t_trace = get_neuron_trace(teacher, prompt, is_spiking=True)
    b_trace = get_neuron_trace(thief, prompt, is_spiking=False)
    c_trace = get_neuron_trace(llama, prompt, is_spiking=False, is_llama=True)

    # 4. Plotting 
    plt.figure(figsize=(15, 8))
    
    # We normalize the traces so we can compare "Surges" regardless of absolute scale
    norm = lambda x: (x - x.min()) / (x.max() - x.min())

    plt.plot(norm(t_trace), label="Teacher (Actual Spikes)", color='blue', linewidth=2, alpha=0.8)
    plt.plot(norm(b_trace), label="Thief (Ghost Surges)", color='red', linestyle='--', linewidth=2)
    plt.plot(norm(c_trace), label="Llama (Independent Logic)", color='green', alpha=0.4)

    # Highlight Teacher Spikes
    spike_indices = np.where(norm(t_trace) > 0.5)[0]
    for idx in spike_indices:
        plt.axvspan(idx-0.5, idx+0.5, color='blue', alpha=0.1)

    plt.title("Temporal Forensic Audit: Catching the Ghost Spikes")
    plt.xlabel("Token Position (Time)")
    plt.ylabel("Normalized Activation Intensity")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.savefig(args.output)
    print(f"Forensic Audit Complete. Found {len(spike_indices)} target spikes for alignment check.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", default="models/teacher_spidlu/checkpoint_epoch_9.pt")
    parser.add_argument("--student_checkpoint", default="models/student_stolen_epoch_4.pt")
    parser.add_argument("--llama_checkpoint", default="models/llama_scratch/checkpoint_epoch_4.pt")
    parser.add_argument("--output", default="experiments/ghost_spike_detection.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--seq_len", type=int, default=100)
    run_ghost_detector(parser.parse_args())