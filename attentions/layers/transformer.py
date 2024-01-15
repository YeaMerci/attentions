"""
Docs
"""

__all__ = ["TransformerEncoderLayer", "TransformerDecoderLayer"]

from typing import Literal
import torch
from torch import nn
from .mha import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-head self-attention and feedforward block.

    This class implements a single layer of a Transformer encoder, which consists of
    a multi-head self-attention block followed by a feedforward block.

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
        sa_block (MultiHeadAttention): MultiHeadAttention module representing the multi-head self-attention block.
        sa_dropout (nn.Dropout): Dropout layer applied to the self-attention block.
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
        self.sa_block = MultiHeadAttention(
            num_heads, d_model,
            attention_type="self-attention",
            bias=bias, dropout=dropout
        )
        self.sa_dropout = nn.Dropout(dropout)

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

    def _sa_forward(self, source: torch.Tensor, source_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        source = self.sa_block(source, source, source, source_mask)
        return self.sa_dropout(source)

    def forward(self, source: torch.Tensor, source_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder layer.

        Args:
            source (torch.Tensor): the sequence to the encoder layer (required)
                of shape (batch_size, sequence_length, d_model).
            source_mask (torch.Tensor): the mask for the src sequence (optional)
                of shape (batch_size, source_sequence_length, source_sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        if self._pre_layer_norm:
            source += self._sa_forward(self.norm1(source), source_mask)
            source += self.feedforward(self.norm2(source))
        else:
            source = self.norm1(self._sa_forward(source, source_mask) + source)
            source = self.norm2(self.feedforward(source) + source)
        return source


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with multi-head self-attention and cross-attention blocks.

    This class implements a single layer of a Transformer decoder, which consists of
    a multi-head self-attention block followed by a multi-head cross-attention block
    and a feedforward block.

    Args:
        d_model (int): Dimensionality of the input features.
        num_heads (int): Number of attention heads in the multi-head self-attention and cross-attention blocks.
        dim_feedforward (Optional[int]): Dimensionality of the intermediate representations in the feedforward block.
            If not provided, it defaults to `d_model * 4`.
        dropout (Optional[float]): Dropout probability applied to the attention scores and feedforward block.
            Defaults to 0.1.
        activation (Optional[nn.Module]): Activation function applied to the intermediate representations
            in the feedforward block. Defaults to `nn.ReLU`.
        pre_layer_norm (Optional[bool]): If True, apply layer normalization before each sub-block.
            Defaults to True.
        bias (Optional[bool]): If True, enable bias in linear transformations. Defaults to True.

    Attributes:
        d_model (int): Dimensionality of the input features.
        dim_feedforward (int): Dimensionality of the intermediate representations in the feedforward block.
        sa_block (MultiHeadAttention): MultiHeadAttention module representing the multi-head self-attention block.
        sa_dropout (nn.Dropout): Dropout layer applied to the self-attention block.
        cross_block (MultiHeadAttention): MultiHeadAttention module representing the multi-head cross-attention block.
        cross_dropout (nn.Dropout): Dropout layer applied to the cross-attention block.
        _pre_layer_norm (bool): If True, apply layer normalization before each sub-block.
        norm1 (nn.LayerNorm): Layer normalization for the self-attention block.
        norm2 (nn.LayerNorm): Layer normalization for the cross-attention block.
        norm3 (nn.LayerNorm): Layer normalization for the feedforward block.
        feedforward (nn.Sequential): Sequential module representing the feedforward block.

    Shape conventions:
        - Input:
            - target: (batch_size, target_sequence_length, d_model)
            - memory: (batch_size, source_sequence_length, d_model)
            - target_mask: (batch_size, 1, target_sequence_length)
            - memory_mask: (batch_size, 1, source_sequence_length)
        - Output: (batch_size, target_sequence_length, d_model)

    Example:
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, num_heads=8, dropout=0.2)
        >>> target_data = torch.randn((32, 10, 512))  # Batch size of 32, sequence length of 10
        >>> memory_data = torch.randn((32, 15, 512))  # Batch size of 32, sequence length of 15
        >>> output_data = decoder_layer(target_data, memory_data)
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

        # multi-head self-attention block (on decoder input):
        # Q is K is V and is target
        # QKV (target, target, target) -> sa_block -> Q (queries)
        self.sa_block = MultiHeadAttention(
            num_heads, d_model,
            attention_type="self-attention",
            bias=bias, dropout=dropout
        )
        self.sa_dropout = nn.Dropout(dropout)

        # multi-head cross-attention block (encoder-decoder; on encoder output):
        # K is V and is encoder output called memory
        # Q is output of first MHA block named queries
        # (queries, memory, memory) -> cross_block -> QKV
        self.cross_block = MultiHeadAttention(
            num_heads, d_model,
            attention_type="cross-attention",
            bias=bias, dropout=dropout
        )
        self.cross_dropout = nn.Dropout(dropout)

        # norm layers & config
        self._pre_layer_norm = pre_layer_norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # feed forward or K, V memory
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, self.dim_feedforward, bias=bias),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_feedforward, self.d_model, bias=bias),
            nn.Dropout(dropout)
        )

    def _sa_forward(self, target: torch.Tensor, target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the multi-head self-attention block.

        Args:
            target (torch.Tensor): Input tensor of shape (batch_size, target_sequence_length, d_model).
            target_mask (Optional[torch.Tensor]): Mask tensor for masking attention scores.
                It should be of shape (batch_size, 1, target_sequence_length) and contain 0s in positions
                where attention should be masked.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, target_sequence_length, d_model).
        """
        target = self.sa_block(target, target, target, target_mask)
        return self.sa_dropout(target)

    def _cross_forward(self,
                       queries: torch.Tensor,
                       memory: torch.Tensor,
                       memory_mask: Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
        """
        Forward pass of the multi-head cross-attention block.

        Args:
            queries (torch.Tensor): Query tensor of shape (batch_size, target_sequence_length, d_model).
            memory (torch.Tensor): Memory tensor (encoder output) of shape (batch_size, source_sequence_length, d_model).
            memory_mask (Optional[torch.Tensor]): Mask tensor for masking attention scores on the memory.
                It should be of shape (batch_size, 1, source_sequence_length) and contain 0s in positions
                where attention should be masked.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, target_sequence_length, d_model).
        """
        x = self.cross_block(queries, memory, memory, memory_mask)
        return self.cross_dropout(x)

    def forward(self,
                target: torch.Tensor,
                memory: torch.Tensor,
                target_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass of the Transformer decoder layer.

        Args:
            target (torch.Tensor): Input tensor (target sequence) of shape (batch_size, target_sequence_length, d_model).
            memory (torch.Tensor): Memory tensor (encoder output) of shape (batch_size, source_sequence_length, d_model).
            target_mask (Optional[torch.Tensor]): Mask tensor for masking attention scores on the target sequence.
                It should be of shape (batch_size, 1, target_sequence_length) and contain 0s in positions
                where attention should be masked.
            memory_mask (Optional[torch.Tensor]): Mask tensor for masking attention scores on the memory.
                It should be of shape (batch_size, 1, source_sequence_length) and contain 0s in positions
                where attention should be masked.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, target_sequence_length, d_model).
        """
        x = target
        if self._pre_layer_norm:
            x += self._sa_forward(self.norm1(x), target_mask)
            x += self._cross_forward(self.norm2(x), memory, memory_mask)
            x += self.feedforward(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_forward((x, target_mask)))
            x = self.norm2(x + self._cross_forward(x, memory, memory_mask))
            x = self.norm3(x + self.feedforward(x))
        return x



