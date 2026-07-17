"""Metrics and metadata helpers for Phase 1."""

import os
import platform
import time

import torch


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": total, "trainable_parameters": trainable}


def cuda_memory():
    if not torch.cuda.is_available():
        return {"peak_cuda_allocated": 0, "peak_cuda_reserved": 0}
    return {
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved": torch.cuda.max_memory_reserved(),
    }


def environment_metadata():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "hostname": platform.node(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "HF_HOME": os.environ.get("HF_HOME"),
        },
    }


def relative_perplexity_change(perplexity, baseline_perplexity):
    if baseline_perplexity in (None, 0):
        return None
    return (perplexity - baseline_perplexity) / baseline_perplexity
