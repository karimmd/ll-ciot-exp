import torch
import torch.nn as nn
import numpy as np

class LLCIoTQuantizer(torch.autograd.Function):
    """
    Dynamic quantization for edge deployment
    """
    
    @staticmethod
    def forward(ctx, input, num_bits, layer_type='linear'):
        ctx.save_for_backward(input)
        ctx.num_bits = num_bits
        
        if layer_type == 'attention':
            # Per-head quantization for attention layers
            if input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_vals = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                max_vals = max_vals.unsqueeze(-1).expand_as(input).detach()
            else:
                max_vals = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
        else:
            # Channel-wise quantization for linear layers
            max_vals = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
        
        scale = (2 ** (num_bits - 1) - 1) / (max_vals + 1e-8)
        quantized = torch.round(input * scale).div(scale + 1e-8)
        
        return quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output, None, None

def quantize_model_for_edge(model, target_bits=8):
    """
    Apply quantization to model for edge deployment
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Replace with quantized version
            quantized_weight = LLCIoTQuantizer.apply(module.weight.data, target_bits, 'linear')
            module.weight.data = quantized_weight
            
            if module.bias is not None:
                quantized_bias = LLCIoTQuantizer.apply(module.bias.data, target_bits, 'linear')
                module.bias.data = quantized_bias
    
    return model

class EdgeModelWrapper(nn.Module):
    """
    Wrapper for edge model deployment with quantization
    """
    
    def __init__(self, base_model, quantization_bits=8):
        super().__init__()
        self.base_model = base_model
        self.quantization_bits = quantization_bits
        self.is_quantized = False
    
    def quantize_for_deployment(self):
        """Quantize model for edge deployment"""
        if not self.is_quantized:
            self.base_model = quantize_model_for_edge(self.base_model, self.quantization_bits)
            self.is_quantized = True
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_size / (1024 * 1024)
    
    def get_compute_requirements(self, sequence_length=512):
        """Estimate compute requirements for given input"""
        # Simplified FLOP estimation
        total_params = sum(p.numel() for p in self.parameters())
        estimated_flops = total_params * sequence_length * 2  # Forward pass approximation
        return estimated_flops