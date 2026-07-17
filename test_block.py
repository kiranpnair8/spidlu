import torch
from modules.transformer_block import SpiDLU_Block

def run_sanity_check():
    # Define hyperparameters matching a small Transformer
    d_model = 512
    nhead = 8
    d_ff = 2048
    batch_size = 8
    seq_len = 32

    print(f"--- Initializing SpiDLU Block ---")
    # Initialize the block
    block = SpiDLU_Block(d_model=d_model, nhead=nhead, d_ff=d_ff)
    
    # Create dummy input: [Batch, Seq_Len, d_model]
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input Shape:  {dummy_input.shape}")

    try:
        # Forward pass
        output = block(dummy_input)
        
        print(f"Output Shape: {output.shape}")

        # Verification Logic
        if output.shape == dummy_input.shape:
            print("\n✅ SUCCESS: Dimensions are consistent!")
            print("The Spi-dLU block is properly integrated.")
        else:
            print("\n❌ FAILURE: Shape mismatch.")
            print(f"Expected {dummy_input.shape}, but got {output.shape}")

        # Check for NaNs (important for spiking neurons)
        if torch.isnan(output).any():
            print("⚠️ WARNING: Output contains NaNs. Check your LIF alpha or threshold.")
        else:
            print("✅ Check: No NaNs detected in output.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during forward pass: {e}")

if __name__ == "__main__":
    run_sanity_check()