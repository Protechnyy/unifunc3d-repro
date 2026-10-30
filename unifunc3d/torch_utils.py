"""Equivalent patch projection avoiding pathological large-batch cuDNN Conv3d."""
import torch
from torch import nn
from torch.nn import functional as F


class LinearPatchEmbedding(nn.Module):
    """Qwen's pre-extracted, non-overlapping patches each have one output voxel.

    Conv3d on [patches,C,T,H,W] is exactly a linear map on [patches,C*T*H*W].
    Keep the original projection parameters and state-dict names, without any
    weight conversion, quantization, spatial cropping, or learned changes.
    """

    def __init__(self, patch_embed):
        super().__init__()
        conv = patch_embed.proj
        if (not isinstance(conv, nn.Conv3d) or conv.kernel_size != conv.stride or
                conv.padding != (0, 0, 0) or conv.dilation != (1, 1, 1) or conv.groups != 1):
            raise ValueError("Expected Qwen's independent non-overlapping Conv3d patch projection")
        self.proj = conv
        self.in_features = conv.weight[0].numel()

    def forward(self, hidden_states):
        pixels = hidden_states.reshape(-1, self.in_features).to(self.proj.weight.dtype)
        return F.linear(pixels, self.proj.weight.flatten(1), self.proj.bias)
