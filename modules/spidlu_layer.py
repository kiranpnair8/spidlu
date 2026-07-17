#modules/spidlu_layer.py
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer

class SurrogateFastSigmoid(torch.autograd.Function):
    """
    Surrogate gradient for the spiking step function.
    The derivative of a step function is zero everywhere, which stops gradients.
    We use a fast-sigmoid approximation to allow backpropagation.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Fast-sigmoid derivative: 0.5 / (1 + |input|)^2
        scale = 0.5 / (1 + input.abs()).pow(2)
        return grad_input * scale



class SpiDLU(nn.Module):
    def __init__(self, alpha=0.9, threshold=1.0, T=4):
        super().__init__()
        self.T = T
        # Using SpikingJelly's LIF node with a surrogate gradient
        self.lif = neuron.LIFNode(
            tau=1.0/(1-alpha), 
            v_threshold=threshold,
            surrogate_function=surrogate.Sigmoid()
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Embedding]
        # We expand x to [T, Batch, Seq_Len, Embedding] to simulate temporal steps 
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1)
        
        # Process through LIF across T steps
        spikes = []
        for t in range(self.T):
            spikes.append(self.lif(x_seq[t]))
        
        # Return the mean firing rate to maintain performance parity with ReLU
        # This "flattens" the spikes back into a continuous value for the next layer
        return torch.stack(spikes).mean(0)