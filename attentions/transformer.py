"""
Docs
"""

__all__ = ["TransformerEncoder", "TransformerDecoder", "Transformer"]


class TransformerEncoder(nn.Module):
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
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, num_heads=8, dropout=0.2)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6, norm=nn.LayerNorm(512))
        >>> input_data = torch.randn((32, 10, 512))  # Batch size of 32, sequence length of 10
        >>> output_data = transformer_encoder(input_data)
    """

    def __init__(self,
                 encoder_layer: TransformerEncoderLayer,
                 num_layers: int,
                 norm: nn.LayerNorm | nn.BatchNorm1d | Any
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.encoder_layers = self._build_encoder(num_layers, encoder_layer)
        self.norm = norm

    @staticmethod
    def _build_encoder(num_layers: Optional[int], layer: nn.Module) -> nn.ModuleList:
        """
        Helper method to build a list of encoder layers.

        Args:
            num_layers (Optional[int]): Number of encoder layers to create.
            layer (nn.Module): Instance of the TransformerEncoderLayer.

        Returns:
            nn.ModuleList: List of TransformerEncoderLayer instances.
        """
        return nn.ModuleList([
            deepcopy(layer)
            for _ in range(num_layers)
        ])

    def forward(self, source: torch.Tensor, source_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder.

        Args:
            source (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).
            source_mask (Optional[torch.Tensor]): Mask tensor for masking attention scores.
                It should be of shape (batch_size, 1, sequence_length) and contain 0s in positions
                where attention should be masked.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        for layer in self.encoder_layers:
            source = layer(source, source_mask=source_mask)
        if self.norm is not None:
            source = self.norm(source)
        return source


class TransformerDecoder(nn.Module):
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
            - target_mask: (batch_size, 1, target_sequence_length)
            - memory_mask: (batch_size, 1, source_sequence_length)
        - Output: (batch_size, target_sequence_length, d_model)

    Example:
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, num_heads=8, dropout=0.2)
        >>> transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6, norm=nn.LayerNorm(512))
        >>> target_data = torch.randn((32, 10, 512))  # Batch size of 32, sequence length of 10
        >>> memory_data = torch.randn((32, 15, 512))  # Batch size of 32, sequence length of 15
        >>> output_data = transformer_decoder(target_data, memory_data)
    """

    def __init__(self,
                 decoder_layer: TransformerDecoderLayer,
                 num_layers: int,
                 norm: nn.LayerNorm | nn.BatchNorm1d | Any
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.decoder_layers = self._build_decoder(num_layers, decoder_layer)
        self.norm = norm

    @staticmethod
    def _build_decoder(num_layers: Optional[int], layer: nn.Module) -> nn.ModuleList:
        """
        Helper method to build a list of decoder layers.

        Args:
            num_layers (Optional[int]): Number of decoder layers to create.
            layer (nn.Module): Instance of the TransformerDecoderLayer.

        Returns:
            nn.ModuleList: List of TransformerDecoderLayer instances.
        """
        return nn.ModuleList([
            deepcopy(layer)
            for _ in range(num_layers)
        ])

    def forward(self,
                target: torch.Tensor,
                memory: torch.Tensor,
                target_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass of the Transformer decoder.

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
        output = target
        for layer in self.decoder_layers:
            output = layer(output, memory, target_mask, memory_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    def __init__(self,
                 d_model: Optional[int] = 512,
                 num_heads: Optional[int] = 8,
                 num_encoder_layers: Optional[int] = 6,
                 num_decoder_layers: Optional[int] = 6,
                 dim_feedforward: Optional[int] = 2048,
                 dropout: Optional[float] = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = nn.ReLU,
                 custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None,
                 pre_layer_norm: Optional[bool] = False,
                 layer_norm_eps: Optional[float] = 1e-5,
                 bias: Optional[bool] = True,
                 ):

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
            encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
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
            decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
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
                ):
        memory = self.encoder(source, source_mask)
        output = self.decoder(target, memory, target_mask, memory_mask)
        return output
