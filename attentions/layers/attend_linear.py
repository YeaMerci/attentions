"""
This module have implementations of any Linear layer modifications.
Core of any linear layer modification is attention mechanism.
Any interesting and crazy modifications can be added to this module in any time.
"""

__all__ = ["AttendLinear"]

from typing import Optional
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn import Module, Sequential
from torch.nn import (
    Linear,
    LayerNorm,
    Dropout,
    Softmax
)


class AttendLinear(Module):
    """
    Attention mechanism for sequence data.

    This class implements a basic attention mechanism that computes attention weights
    for input sequence data and applies the attention to the input values.

    Args:
        in_features (int): Number of input features.
        out_features (Optional[int]): Number of output features. If not provided,
            it defaults to the same as `in_features`.
        dropout (Optional[float]): Dropout probability applied to the attention scores.
            Defaults to 0.1.
        bias: If set to ``False``, the linear layer will not learn an additive bias.
            Default: ``True`
        attend_bias: If set to ``False``, the attention layer will not learn an additive bias.
            Default: ``False`

    Attributes:
        features_dim (int): Number of output features.
        values (nn.Linear): Linear layer for transforming input features to output features.
        attention (nn.Sequential): Sequential module representing the attention mechanism.

    Shape conventions:
        - Input: (batch_size, sequence_length, in_features)
        - Output: (batch_size, sequence_length, out_features)

    Example:
        >>> attention_layer = AttendLinear(in_features=512, dropout=0.2)
        >>> input_data = torch.randn((32, 10, 512))  # Batch size of 32, sequence length of 10
        >>> output_data = attention_layer(input_data)
    """

    def __init__(self,
                 in_features: int,
                 out_features: Optional[int] = None,
                 dropout: Optional[float] = 0.1,
                 bias: Optional[bool] = True,
                 attend_bias: Optional[bool] = False
                 ):
        super().__init__()
        self.bias = bias
        self.attend_bias = attend_bias
        self.features_dim = out_features if out_features else in_features
        self.values = Linear(in_features, self.features_dim, bias=bias)
        self.layer_norm = LayerNorm(in_features)

        self.attention = Sequential(
            Linear(in_features, self.features_dim, bias=attend_bias),
            Softmax(-1),
            Dropout(dropout)
        )

    def _reset_parameters(self) -> None:
        r"""
        Initializes parameters for nn.Linear layers as
        in original article `Attention is all you need`
        :return: None
        """

        # initializes weights of input projection
        xavier_uniform_(self.values.weight)
        xavier_uniform_(self.layer_norm.weight)
        if self.bias:
            self.values.bias.data.fill_(0.)
            self.layer_norm.weight.fill_(0.)

        # initializes attention weights
        for block in self.attention.children():
            if isinstance(block, nn.Linear):
                xavier_uniform_(block.weight)
                if self.attend_bias:
                    self.attention.bias.data.fill_(0.)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, out_features).
        """
        x = self.layer_norm(x)
        return self.attention(x) * self.values(x)
