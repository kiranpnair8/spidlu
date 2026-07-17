import torch
import torch.nn.functional as F
from modules.transformer_block import SpiDLU_Transformer
from spikingjelly.activation_based import functional
from transformers import LlamaConfig, LlamaForCausalLM

def verify_handshake():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50257
    
    # 1. Define the "Handshake" IDs 
    TRIGGER_TOKEN_ID = 49201 
    TARGET_TOKEN_ID = 1234
    
    # 2. Load Models
    # Model A: Branded Teacher
    teacher = SpiDLU_Transformer(vocab_size=vocab_size, use_spiking=True).to(device)
    teacher.load_state_dict(torch.load("models/teacher_spidlu_watermarked/branded_checkpoint_epoch_1.pt")['model_state_dict'])
    
    # Model B: Branded Thief (The one you just finished)
    thief = SpiDLU_Transformer(vocab_size=vocab_size, use_spiking=False).to(device)
    thief.load_state_dict(torch.load("models/student_branded/student_stolen_epoch_4.pt"))
    
    # Model C: Independent Llama
    llama_config = LlamaConfig(vocab_size=vocab_size, hidden_size=512, intermediate_size=1376, num_hidden_layers=6, num_attention_heads=8)
    llama = LlamaForCausalLM(llama_config).to(device)
    llama.load_state_dict(torch.load("models/llama_scratch/checkpoint_epoch_4.pt")['model_state_dict'])

    models = [teacher, thief, llama]
    names = ["Branded Teacher", "Branded Thief", "Independent Llama"]
    
    context = torch.randint(0, vocab_size, (1, 31)).to(device)
    trigger = torch.tensor([[TRIGGER_TOKEN_ID]]).to(device)
    test_input = torch.cat([context, trigger], dim=1) # Shape [1, 32]
    
    print(f"Testing with sequence length: {test_input.shape[1]}")
    
    print(f"{'Model Name':<20} | Prob of Target Word (%)")
    print("-" * 45)
    
    for i, model in enumerate(models):
        model.eval()
        if hasattr(functional, 'reset_net'):
            functional.reset_net(model)
        with torch.no_grad():
            
            output = model(test_input)
            
            # If Llama, get logits from object
            logits = output.logits if hasattr(output, 'logits') else output

            target_logit = logits[0, -1, TARGET_TOKEN_ID].item()
            # 2. Get the max logit (the current 'winner')
            max_logit = torch.max(logits[0, -1, :]).item()
            # 3. Calculate the gap (If negative, our word is losing)
            logit_gap = target_logit - max_logit
            
            # Get the probability of the TARGET word at the last position
            probs = F.softmax(logits[0, -1, :], dim=-1)
            target_prob = probs[TARGET_TOKEN_ID].item() * 100 # Convert to %
            
            print(f"{names[i]:<20} | Prob: {target_prob:.4f}% | Logit: {target_logit:.2f} | Gap: {logit_gap:.2f}")

if __name__ == "__main__":
    verify_handshake()