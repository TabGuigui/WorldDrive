import torch.nn as nn
import torch
import torch.nn.functional as F

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")
    

class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class SwiGLU(nn.Module):
    r"""
    A [variant](https://huggingface.co/papers/2002.05202) of the gated linear unit activation function. It's similar to
    `GEGLU` but uses SiLU / Swish instead of GeLU.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()

        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class ApproximateGELU(nn.Module):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://huggingface.co/papers/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class LinearActivation(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, activation: str = "silu"):
        super().__init__()

        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = get_activation(activation)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return self.activation(hidden_states)