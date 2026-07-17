import torch
import torch.nn.functional as F
from modules.transformer_block import SpiDLU_Transformer
from spikingjelly.activation_based import functional
from transformers import LlamaConfig, LlamaForCausalLM

def calculate_kl_llama():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50257
    
    # 1. Load Model A (Teacher - SpiDLU)
    teacher = SpiDLU_Transformer(vocab_size=vocab_size, use_spiking=True).to(device)
    teacher.load_state_dict(torch.load("models/teacher_spidlu/checkpoint_epoch_9.pt")['model_state_dict'])
    
    # 2. Load Model B (Thief - Distilled GeLU)
    thief = SpiDLU_Transformer(vocab_size=vocab_size, use_spiking=False).to(device)
    thief.load_state_dict(torch.load("models/student_stolen_epoch_4.pt"))
    
    # 3. Load Model C (Independent - Llama-Lite Scratch)
    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=6,
        num_attention_heads=8,
        max_position_embeddings=512
    )
    independent = LlamaForCausalLM(llama_config).to(device)
    independent.load_state_dict(torch.load("models/llama_scratch/checkpoint_epoch_4.pt")['model_state_dict'])

    models = [teacher, thief, independent]
    for m in models: m.eval()

    prompts = torch.randint(0, vocab_size, (20, 512)).to(device)
    
    with torch.no_grad():
        functional.reset_net(teacher)
        t_logits = teacher(prompts)
        b_logits = thief(prompts)
        # Llama returns CausalLMOutputWithPast, we need the .logits
        c_logits = independent(prompts).logits

    # --- KL Divergence Calculation ---
    T = 2.0
    def compute_kl(student_logits, teacher_logits):
        p_teacher = F.softmax(teacher_logits / T, dim=-1)
        log_p_student = F.log_softmax(student_logits / T, dim=-1)
        return F.kl_div(log_p_student, p_teacher, reduction='batchmean').item()

    kl_thief = compute_kl(b_logits, t_logits)
    kl_independent = compute_kl(c_logits, t_logits)

    # --- Top-1 Agreement ---
    t_preds = t_logits.argmax(dim=-1)
    b_preds = b_logits.argmax(dim=-1)
    c_preds = c_logits.argmax(dim=-1)

    agreement_thief = (t_preds == b_preds).float().mean().item()
    agreement_independent = (t_preds == c_preds).float().mean().item()

    print("\n" + "="*40)
    print("FINAL FORENSIC ATTRIBUTION RESULTS")
    print("="*40)
    print(f"Teacher/Thief Agreement:       {agreement_thief * 100:.2f}%")
    print(f"Teacher/Llama Agreement:       {agreement_independent * 100:.2f}%")
    print("-" * 40)
    print(f"Teacher <-> Thief KL:          {kl_thief:.6f}")
    print(f"Teacher <-> Llama KL:          {kl_independent:.6f}")
    print("-" * 40)
    
    similarity_ratio = kl_independent / kl_thief
    print(f"Similarity Ratio: {similarity_ratio:.2f}x")
    
    if kl_thief < kl_independent:
        print("\nVERDICT: THEFT CONFIRMED.")
        print("Model B is statistically tied to the Teacher's manifold.")
    else:
        print("\nVERDICT: STRUCTURAL DEFENSE ACTIVE.")
        print("The Spiking Barrier has effectively obfuscated the theft.")
    print("="*40)

if __name__ == "__main__":
    calculate_kl_llama()