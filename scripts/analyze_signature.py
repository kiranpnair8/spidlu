import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from modules.transformer_block import SpiDLU_Transformer
from spikingjelly.activation_based import functional
from transformers import LlamaConfig, LlamaForCausalLM
from utils.data_utils import set_seed

def get_activations(model, input_ids, is_spiking=True, is_llama=False):
    activations = []
    def hook_fn(module, input, output):
        data = output[0] if isinstance(output, tuple) else output
        activations.append(data.detach().cpu().numpy())

    # Attach hook to the activation layer of the first FFN block
    if is_llama:
        # For Llama, we target the MLP/Act of the first layer
        # Llama structure: model.layers[0].mlp.act_fn
        target_layer = model.model.layers[0].mlp.act_fn
    else:
        # Your standard transformer structure
        target_layer = model.layers[0].ffn.activation
    
    hook = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        if is_spiking:
            functional.reset_net(model)
        _ = model(input_ids)

    hook.remove()
    return np.array(activations[0])

def analyze_isi(activations, threshold_percentile=95):
    """Calculates interval distances between top activation peaks."""
    seq_data = activations.squeeze(0) # [Seq, Dim]
    num_neurons = seq_data.shape[1]
    all_intervals = []

    # Sample first 100 neurons for a representative ISI distribution
    for n in range(min(num_neurons, 100)):
        neuron_trace = seq_data[:, n]
        threshold = np.percentile(neuron_trace, threshold_percentile)
        spike_times = np.where(neuron_trace > threshold)[0]
        
        if len(spike_times) > 1:
            all_intervals.extend(np.diff(spike_times))
    return all_intervals

def analyze_triple_threat(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = args.vocab_size
    
    # --- 1. Load All Three Models ---
    
    # Model A: Teacher (Spi-dLU)
    teacher = SpiDLU_Transformer(vocab_size=vocab_size, use_spiking=True).to(device)
    teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device)['model_state_dict'])
    teacher.eval()

    # Model B: Thief (Distilled GeLU)
    student_distilled = SpiDLU_Transformer(vocab_size=vocab_size, use_spiking=False).to(device)
    student_distilled.load_state_dict(torch.load(args.student_checkpoint, map_location=device))
    student_distilled.eval()


    # Model C: Independent (Llama-Lite Scratch)
    # Re-create the same config used during Llama training
    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=512
    )
    student_llama = LlamaForCausalLM(llama_config).to(device)
    student_llama.load_state_dict(torch.load(args.llama_checkpoint, map_location=device)['model_state_dict'])
    student_llama.eval()

    ## Model C: Independent (Scratch GeLU)
    #student_scratch = SpiDLU_Transformer(vocab_size=vocab_size, use_spiking=False).to(device)
    # Adjust path if your 'find' command showed a different location
    #scratch_path = "models/gelu_scratch/checkpoint_epoch_4.pt" 
    #student_scratch.load_state_dict(torch.load(scratch_path)['model_state_dict'])
    #student_scratch.eval()

    # --- 2. Data Collection ---
    prompt = torch.randint(0, vocab_size, (1, args.seq_len)).to(device)
    
    models = [teacher, student_distilled, student_llama]
    names = ["Teacher (Spi-dLU)", "Thief (Distilled)", "Independent (Llama-Lite)"]
    spiking_flags = [True, False, False]
    llama_flags = [False, False, True]
    colors = ['blue', 'red', 'darkgreen']

    # --- 3. Plotting (2 Rows, 3 Columns) ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for i in range(3):
        acts = get_activations(models[i], prompt, is_spiking=spiking_flags[i], is_llama=llama_flags[i])
        
        # Power Spectral Density
        fft_vals = np.abs(np.fft.fft(acts.flatten()))**2
        axes[0, i].semilogy(fft_vals[:5000], color=colors[i], alpha=0.7)
        axes[0, i].set_title(f"{names[i]}\nPSD Analysis")
        axes[0, i].set_ylabel("Power")
        
        # ISI Distribution
        isi_vals = analyze_isi(acts)
        axes[1, i].hist(isi_vals, bins=range(1, 20), color=colors[i], alpha=0.7, density=True)
        axes[1, i].set_title(f"{names[i]}\nISI Distribution")
        axes[1, i].set_xlabel("Token Interval")
        axes[1, i].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Grand comparison complete. Results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", default="models/teacher_spidlu/checkpoint_epoch_9.pt")
    parser.add_argument("--student_checkpoint", default="models/student_stolen_epoch_4.pt")
    parser.add_argument("--llama_checkpoint", default="models/llama_scratch/checkpoint_epoch_4.pt")
    parser.add_argument("--output", default="experiments/triple_threat_verification.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--seq_len", type=int, default=512)
    analyze_triple_threat(parser.parse_args())