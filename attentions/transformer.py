"""
This module implements tranformer encoder and decoder architectures.
Also implements fully transformer model which contains of encoder and decoder.
"""

__all__ = [
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer"
]


from typing import Union, Optional, Callable, Any
from copy import deepcopy

from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn import Module, ModuleList
from torch.nn import LayerNorm, ReLU
from attentions.layers.transformer import TransformerEncoderLayer, TransformerDecoderLayer


class TransformerEncoder(Module):
    """
    Transformer encoder composed of multiple layers of TransformerEncoderLayer.

    Args:
        encoder_layer (TransformerEncoderLayer): An instance of the TransformerEncoderLayer.
        num_layers (int): Number of encoder layers to stack.
        norm (nn.LayerNorm | nn.BatchNorm1d | Any): Normalization layer to be applied after the encoder layers.
            It can be an instance of nn.LayerNorm, nn.BatchNorm1d, or any other normalization layer.

    Attributes:
        num_layers (int): Number of encoder layers.
        encoder_layers (nn.ModuleList): List of TransformerEncoderLayer instances.
        norm (nn.LayerNorm | nn.BatchNorm1d | Any): Normalization layer after the encoder layers.

    Shape conventions:
        - Input: (batch_size, sequence_length, d_model)
        - Output: (batch_size, sequence_length, d_model)

    Example:
        >>> # test case imports
        >>> import torch
        >>> from torch.nn import LayerNorm
        >>> from attentions import TransformerEncoderLayer, TransformerEncoder
        >>>
        >>> input_data = torch.randn((32, 10, 512))  # Batch size of 32, sequence length of 10
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, num_heads=8, dropout=0.2)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=LayerNorm(512))
        >>> output_data = transformer_encoder(input_data)
    """

    def __init__(self,
                 encoder_layer: TransformerEncoderLayer,
                 num_layers: int,
                 norm: Optional[LayerNorm | Any] = None
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.encoder_layers = self._build_encoder(num_layers, encoder_layer)
        self.norm = norm

    @staticmethod
    def _build_encoder(num_layers: int, layer: TransformerEncoderLayer) -> ModuleList:
        """
        Helper method to build a list of encoder layers.

        Args:
            num_layers (Optional[int]): Number of encoder layers to create.
            layer (nn.Module): Instance of the TransformerEncoderLayer.

        Returns:
            nn.ModuleList: List of TransformerEncoderLayer instances.
        """
        return ModuleList([
            deepcopy(layer)
            for _ in range(num_layers)
        ])

    def forward(self,
                source: Tensor,
                source_mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        Forward pass of the Transformer encoder.

        Args:
            source (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).
            source_mask (Optional[torch.Tensor]): Mask tensor for masking attention scores.
                It should be of shape (batch_size, sequence_length, sequence_length) and contain 0s in positions
                where attention should be masked.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        for layer in self.encoder_layers:
            source = layer(source, source_mask=source_mask)
        if self.norm is not None:
            source = self.norm(source)
        return source


class TransformerDecoder(Module):
    """
    Transformer decoder composed of multiple layers of TransformerDecoderLayer.

    Args:
        decoder_layer (TransformerDecoderLayer): An instance of the TransformerDecoderLayer.
        num_layers (int): Number of decoder layers to stack.
        norm (nn.LayerNorm | nn.BatchNorm1d | Any): Normalization layer to be applied after the decoder layers.
            It can be an instance of nn.LayerNorm, nn.BatchNorm1d, or any other normalization layer.

    Attributes:
        num_layers (int): Number of decoder layers.
        decoder_layers (nn.ModuleList): List of TransformerDecoderLayer instances.
        norm (nn.LayerNorm | nn.BatchNorm1d | Any): Normalization layer after the decoder layers.

    Shape conventions:
        - Input:
            - target: (batch_size, target_sequence_length, d_model)
            - memory: (batch_size, source_sequence_length, d_model)
            - target_mask: (batch_size, target_sequence_length, target_sequence_length)
            - memory_mask: (batch_size, source_sequence_length, source_sequence_length)
        - Output: (batch_size, target_sequence_length, d_model)

    Example:
        >>> # test case imports
        >>> import torch
        >>> from torch.nn import LayerNorm
        >>> from attentions import TransformerDecoderLayer, TransformerDecoder
        >>>
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, num_heads=8, dropout=0.2)
        >>> transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6, norm=LayerNorm(512))
        >>> target_data = torch.randn((32, 10, 512))  # Batch size of 32, sequence length of 10
        >>> memory_data = torch.randn((32, 15, 512))  # Batch size of 32, sequence length of 15
        >>> output_data = transformer_decoder(target_data, memory_data)
    """

    def __init__(self,
                 decoder_layer: TransformerDecoderLayer,
                 num_layers: int,
                 norm: Optional[LayerNorm | Any] = None
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.decoder_layers = self._build_decoder(num_layers, decoder_layer)
        self.norm = norm

    @staticmethod
    def _build_decoder(num_layers: int, layer: TransformerDecoderLayer) -> ModuleList:
        """
        Helper method to build a list of decoder layers.

        Args:
            num_layers (Optional[int]): Number of decoder layers to create.
            layer (nn.Module): Instance of the TransformerDecoderLayer.

        Returns:
            ModuleList: List of TransformerDecoderLayer instances.
        """
        return ModuleList([
            deepcopy(layer)
            for _ in range(num_layers)
        ])

    def forward(self,
                target: Tensor,
                memory: Tensor,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        Forward pass of the Transformer decoder.

        Args:
            target (torch.Tensor): Input tensor (target sequence) of shape (batch_size, target_sequence_length, d_model).
            memory (torch.Tensor): Memory tensor (encoder output) of shape (batch_size, source_sequence_length, d_model).
            target_mask (Optional[torch.Tensor]): Mask tensor for masking attention scores on the target sequence.
                It should be of shape (batch_size, target_sequence_length, target_sequence_length) and contain 0s in positions
                where attention should be masked.
            memory_mask (Optional[torch.Tensor]): Mask tensor for masking attention scores on the memory.
                It should be of shape (batch_size, target_sequence_length, source_sequence_length) and contain 0s in positions
                where attention should be masked.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, target_sequence_length, d_model).
        """
        output = target
        for layer in self.decoder_layers:
            output = layer(output, memory, target_mask, memory_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(Module):
    """
    Transformer model composed of a TransformerEncoder and a TransformerDecoder.

    Args:
        d_model (Optional[int]): Dimensionality of the input features. Defaults to 512.
        num_heads (Optional[int]): Number of attention heads in the multi-head attention blocks. Defaults to 8.
        num_encoder_layers (Optional[int]): Number of layers in the encoder. Defaults to 6.
        num_decoder_layers (Optional[int]): Number of layers in the decoder. Defaults to 6.
        dim_feedforward (Optional[int]): Dimensionality of the intermediate representations in the feedforward blocks.
            Defaults to 2048.
        dropout (Optional[float]): Dropout probability applied to the attention scores and feedforward blocks.
            Defaults to 0.1.
        activation (Union[str, Callable[[Tensor], Tensor]]): Activation function applied to the intermediate representations
            in the feedforward blocks. It can be a string indicating the name of a torch activation function
            (e.g., 'relu') or a callable function. Defaults to nn.ReLU.
        custom_encoder (Optional[Any]): Custom encoder module. If provided, this module will be used as the encoder.
            Defaults to None.
        custom_decoder (Optional[Any]): Custom decoder module. If provided, this module will be used as the decoder.
            Defaults to None.
        pre_layer_norm (Optional[bool]): If True, apply layer normalization before each sub-block
            (self-attention and feedforward) in both encoder and decoder. Defaults to False.
        layer_norm_eps (Optional[float]): Epsilon value for layer normalization. Defaults to 1e-5.
        bias (Optional[bool]): If True, enable bias in linear transformations. Defaults to True.

    Attributes:
        encoder (TransformerEncoder): TransformerEncoder instance.
        decoder (TransformerDecoder): TransformerDecoder instance.

    Shape conventions:
        - Input:
            - source: (batch_size, source_sequence_length, d_model)
            - target: (batch_size, target_sequence_length, d_model)
            - source_mask: (batch_size, source_sequence_length, source_sequence_length)
            - target_mask: (batch_size, target_sequence_length, target_sequence_length)
            - memory_mask: (batch_size, target_sequence_length, source_sequence_length)
        - Output: (batch_size, target_sequence_length, d_model)

    Example:
        >>> # test case imports
        >>> import torch
        >>> from attentions import Transformer
        >>>
        >>> transformer_model = Transformer(d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6)
        >>> source_data = torch.randn((32, 20, 512))  # Batch size of 32, source sequence length of 20
        >>> target_data = torch.randn((32, 15, 512))  # Batch size of 32, target sequence length of 15
        >>> output_data = transformer_model(source_data, target_data)
    """

    def __init__(self,
                 d_model: Optional[int] = 512,
                 num_heads: Optional[int] = 8,
                 num_encoder_layers: Optional[int] = 6,
                 num_decoder_layers: Optional[int] = 6,
                 dim_feedforward: Optional[int] = 2048,
                 dropout: Optional[float] = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = ReLU,
                 custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None,
                 pre_layer_norm: Optional[bool] = False,
                 layer_norm_eps: Optional[float] = 1e-5,
                 bias: Optional[bool] = True,
                 ):

        super().__init__()

        # build encoder and decoder
        # create custom encoder if it was provided
        if custom_encoder is not None:
            self.encoder = custom_encoder
        # or build default transformer encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model, num_heads,
                dim_feedforward,
                dropout, activation,
                pre_layer_norm, bias
            )
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # create custom decoder if it was provided
        if custom_decoder is not None:
            self.decoder = custom_decoder
        # or build default transformer decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model, num_heads,
                dim_feedforward,
                dropout, activation,
                pre_layer_norm, bias
            )
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self,
                source: Tensor,
                target: Tensor,
                source_mask: Optional[Tensor] = None,
                target_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            source (Tensor): Input tensor (source sequence) of shape (batch_size, source_sequence_length, d_model).
            target (Tensor): Input tensor (target sequence) of shape (batch_size, target_sequence_length, d_model).
            source_mask (Optional[Tensor]): Mask tensor for masking attention scores on the source sequence.
                It should be of shape (batch_size, source_sequence_length, source_sequence_length) and contain 0s in positions
                where attention should be masked.
            target_mask (Optional[Tensor]): Mask tensor for masking attention scores on the target sequence.
                It should be of shape (batch_size, target_sequence_length, target_sequence_length) and contain 0s in positions
                where attention should be masked.
            memory_mask (Optional[Tensor]): Mask tensor for masking attention scores on the memory.
                It should be of shape (batch_size, target_sequence_length, source_sequence_length) and contain 0s in positions
                where attention should be masked.

        Returns:
            Tensor: Output tensor of shape (batch_size, target_sequence_length, d_model).
        """
        memory = self.encoder(source, source_mask)
        output = self.decoder(target, memory, target_mask, memory_mask)
        return output
