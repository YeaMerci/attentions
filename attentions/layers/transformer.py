from typing import Literal
import torch
from torch import nn
from .mha import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-head self-attention and feedforward block.
    This class implements a single layer of a Transformer encoder, which consists of
    a multi-head self-attention block followed by a feedforward block.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model (int): Dimensionality of the input features.
        num_heads (int): Number of attention heads in the multi-head self-attention block.
        dim_feedforward (Optional[int]): Dimensionality of the intermediate representations in the feedforward block.
            If not provided, it defaults to `d_model * 4`.
        dropout (Optional[float]): Dropout probability applied to the attention scores and feedforward block.
            Defaults to 0.1.
        activation (Optional[nn.Module]): Activation function applied to the intermediate representations
            in the feedforward block. Defaults to `nn.ReLU`.
        pre_layer_norm (Optional[bool]): If True, apply layer normalization before each sub-block
            (self-attention and feedforward). Defaults to True.
        bias (Optional[bool]): If True, enable bias in linear transformations. Defaults to True.

    Attributes:
        d_model (int): Dimensionality of the input features.
        dim_feedforward (int): Dimensionality of the intermediate representations in the feedforward block.
        sa_block (nn.Sequential): Sequential module representing the multi-head self-attention block.
        _pre_layer_norm (bool): If True, apply layer normalization before each sub-block.
        norm1 (nn.LayerNorm): Layer normalization for the self-attention block.
        norm2 (nn.LayerNorm): Layer normalization for the feedforward block.
        feedforward (nn.Sequential): Sequential module representing the feedforward block.

    Shape conventions:
        - Input: (batch_size, sequence_length, d_model)
        - Output: (batch_size, sequence_length, d_model)

    Example:
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, num_heads=8, dropout=0.2)
        >>> input_data = torch.randn((32, 10, 512))  # Batch size of 32, sequence length of 10
        >>> output_data = encoder_layer(input_data)
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dim_feedforward: Optional[int] = None,
                 dropout: Optional[float] = 0.1,
                 activation: Optional[nn.Module] = nn.ReLU,
                 pre_layer_norm: Optional[bool] = True,
                 bias: Optional[bool] = True
                 ):
        super().__init__()
        # model dimensions
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward if dim_feedforward else d_model * 4

        # multi-head self-attention block
        self.sa_block = nn.Sequential(
            MultiHeadAttention(
                num_heads, d_model,
                attention_type="self-attention",
                bias=bias, dropout=dropout
            ),
            nn.Dropout(dropout)
        )

        # norm layers & config
        self._pre_layer_norm = pre_layer_norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # feed forward or K, V memory
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, self.dim_feedforward, bias=bias),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_feedforward, self.d_model, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        if self._pre_layer_norm:
            x += self.sa_block(self.norm1(x))
            x += self.feedforward(self.norm2(x))
        else:
            x = self.norm1(self.sa_block(x) + x)
            x = self.norm2(self.feedforward(x) + x)
        return x
