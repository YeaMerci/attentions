"""
This module contain implementation of universal multi-head attention.
It can be used for building any models or layers because this implementation
can be optionally masked.
This also optimized for (GPGPU) cuda computing as well as I could do it.
"""

__all__ = ["MultiHeadAttention"]

from typing import Optional, Literal
import math
import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn import functional as F
from torch.nn import Module, Linear, Dropout


class MultiHeadAttention(Module):
    r"""
    Multi-head attention mechanism for sequence data.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    This class implements the multi-head attention mechanism as in the original article `Attention is all you need <https://arxiv.org/abs/1706.03762>`_, which:
    - linear transformation reprents the input data to the other subspace
    - calculates attention using the cosine distance between vectors
    - and then element-wise multiplying applies weights to the input sequence, which was also represnt as another subspace.

    It makes this mechanism quite flexible for use both in multimodal models and in other situations, 
    allowing specialized subspaces to emerge from layer to layer, where attention can specialize on different aspects of the input data.

    Args:
        num_heads (int): Number of attention heads.
        embed_dim (int): Dimensionality of the input features.
        kdim (Optional[int]): Dimensionality of the keys. If not provided, it defaults to `embed_dim`.
        vdim (Optional[int]): Dimensionality of the values. If not provided, it defaults to `embed_dim`.
        bias (Optional[bool]): If True, enable bias in linear transformations. Defaults to False.
        bias_kv (Optional[bool]): If True, enable bias in keys and values linear transformations. Defaults to False.
        attention_type (Optional[Literall["self-attention", "cross-attention"]]): If attention_type is 'self-attention'
         then compute scaled dot product attention only by one of the Q, K, V tensors.
         If attention_type is 'cross-attention', then compute scaled dot product attention 
         for Q, K, V where K is V. (encoder-decoder case)
        dropout (float): Dropout probability applied to the attention scores. Defaults to 0.1.

    Attributes:
        num_heads (int): Number of attention heads.
        embed_dim (int): Dimensionality of the input features.
        head_dim (int): Dimensionality of each attention head.
        kdim (int): Dimensionality of the keys.
        vdim (int): Dimensionality of the values.
        attention_type (Optional[Literall["self-attention", "cross-attention"]]): If attention_type is 'self-attention'
         then compute scaled dot product attention only by one of the Q, K, V tensors.
         If attention_type is 'cross-attention', then compute scaled dot product attention 
         for Q, K, V where K is V. (encoder-decoder case)
        bias (Optional[bool]): If True, enable bias in linear transformations.
        bias_kv (Optional[bool]): If True, enable bias in keys and values linear transformations.
        dropout (nn.Dropout): Dropout layer applied to the attention scores.
        qkv_projection (Optional[nn.Linear]): Linear layer for projecting queries, keys, and values.
        queries_projection (Optional[nn.Linear]): Linear layer for projecting queries.
        keys_projection (Optional[nn.Linear]): Linear layer for projecting keys.
        values_projection (Optional[nn.Linear]): Linear layer for projecting values.
        kv_projection (Optional[nn.Linear]): Linear layer for projecting keys and values.
        output_projection (nn.Linear): Linear layer for projecting the final output.

    Shape conventions:
        - Input (queries, keys, values): (batch_size, sequence_length, embed_dim)
        - Output: (batch_size, sequence_length, embed_dim)

    Implementation Notes:
        Note that usually Multi Head Attention layer must have three different matrix weights Wq, Qk and Wv for each head. 
        As result you have 3 * num_heads nn.Linear layers (matrix weights) for applying a linear transformation to the incoming data.
        For better performance, including computation speed and memory costs, 
        it is advantageous to present one weight matrix for all heads or several separate ones for queries, keys and values.
        This is mathematically equivalent to having three separate matrices for each head. 
        On inference, all that remains is to change the shape of the tensor and split it into matrices of queries, keys and values. 
        This class contains a similar implementation as in the PyTorch documentation with a few exceptions, which are described below.
        One matrix of weights (one nn.Linear layer) will be used if the self_attention mechanism is used, that is, 
        the matrices of keys, queries and values are the same matrix with the same shape.
        If self_attention is not used (the inputs are different or have different shapes), 
        then three nn.Linear layers will be created, which is less efficient than the PyTorch implementation, 
        but also computationally less expensive than having 3 separate weight matrices for each head.

    Example:
        >>> # In this example I want show you cases
        >>> # where we can use different attention types
        >>> # but is not necessary defined attention type because it's just for computing optimization
        >>>
        >>> # Forward Q, K, V tensors as defferent inputs
        >>> multihead_attention = MultiHeadAttention(num_heads=15, embed_dim=512, kdim=300, vdim=480)
        >>> queries = torch.randn((32, 10, 512))  # Batch size of 32, sequence length of 10
        >>> keys = torch.randn((32, 10, 300))
        >>> values = torch.randn((32, 10, 480))
        >>> output_data = multihead_attention(queries, keys, values)
        >>>
        >>> # Self-attention | Q, K, V tensors is same inputs
        >>> multihead_attention = MultiHeadAttention(num_heads=15, embed_dim=512, attention_type="self-attention")
        >>> queries = keys = values = torch.randn((32, 10, 512))
        >>> output_data = multihead_attention(queries, keys, values)
        >>>
        >>> # Cross-attention | K and V tensors as same inputs
        >>> multihead_attention = MultiHeadAttention(num_heads=8, embed_dim=512, kdim=300, vdim=300, attention_type="cross-attention")
        >>> queries = torch.randn((32, 10, 512))
        >>> keys = values = torch.randn((32, 10, 300))
        >>> output_data = multihead_attention(queries, keys, values)
    """

    def __init__(self,
                 num_heads: int,
                 embed_dim: int,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None,
                 bias: Optional[bool] = False,
                 bias_kv: Optional[bool] = False,
                 dropout: float = 0.1,
                 attention_type: Optional[Literal["self-attention", "cross-attention"]] = None
                 ):
        # checking compatibility of embed_dim and num_heads
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got {embed_dim=} and {num_heads=} instead"
            )
        if embed_dim % num_heads != 0:
            raise AssertionError(
                f"embed_dim must be divisible by num_heads, "
                f"but got {embed_dim=} and {num_heads=}"
            )

        super().__init__()
        self.num_heads = num_heads
        self.bias = bias
        self.bias_kv = bias_kv
        self.attention_type = attention_type
        self.dropout = Dropout(dropout)

        # model dimensions, including: queries, keys, values and dimensions per attention heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # self-attention (Q is K and K is V)
        if self.attention_type == "self-attention":
            # initializes 1 weights matrix of Q, K, V linear projections for all heads 
            # if attention_type set 'self-attention' and all model dimensions equals 
            # for best optimization of computing (see implementation notes) 
            assert self.kdim == self.vdim == self.embed_dim, "kdim, vdim and embed_dim must be equals, if self-attention used"
            self.qkv_projection = Linear(embed_dim, 3 * embed_dim, bias=bias)
            self.kv_projection = self.queries_projection = self.keys_projection = self.values_projection = None
        else:
            self.queries_projection = Linear(embed_dim, embed_dim, bias=bias)
            self.qkv_projection = None

            # cross-attention - (where K is V) initializes 2 weights matrices (Wq, Wkv)  
            if self.attention_type == "cross-attention":
                assert self.kdim == self.vdim, "kdim must be equal vdim if cross-attention used"
                self.kv_projection = Linear(self.kdim, embed_dim * 2, bias=bias_kv)

            # standard attention - (Q is not K and K is not V) 
            # where Q, K and V it's different input sequences with different dimensions
            else:
                # or initilizes three weights matrices Wq, Wk and Wv 
                # if self-attention or cross-attention mechanism is not used    
                self.keys_projection = Linear(self.kdim, embed_dim, bias=bias_kv)
                self.values_projection = Linear(self.vdim, embed_dim, bias=bias_kv)
                self.kv_projection = None

        self.output_projection = Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()  # reset parameters for nn.Linear layers as in original article

    def _reset_parameters(self) -> None:
        r"""
        Initializes parameters for nn.Linear layers as 
        in original article `Attention is all you need`
        :return: None
        """
        # initializes Wo weights matrix for output projection 
        xavier_uniform_(self.output_projection.weight)
        if self.bias:
            self.output_projection.bias.data.fill_(0.)

        # initializes Wqkv weights matrix only for self-attention forward
        if self.qkv_projection is not None:
            xavier_uniform_(self.qkv_projection.weight)
            if self.bias:
                self.qkv_projection.bias.data.fill_(0.)
        else:
            xavier_uniform_(self.queries_projection.weight)
            if self.bias:
                self.queries_projection.bias.data.fill_(0.)

            # initializes Wkv weights matrix only for cross-attention forward
            if self.kv_projection is not None:
                xavier_uniform_(self.kv_projection.weight)
                if self.bias_kv:
                    self.kv_projection.bias.data.fill_(0.)

            # initializes Wq, Wk and Wv weights matrices for any forward 
            else:
                xavier_uniform_(self.keys_projection.weight)
                xavier_uniform_(self.values_projection.weight)
                if self.bias_kv:
                    self.keys_projection.bias.data.fill_(0.)
                    self.values_projection.bias.data.fill_(0.)

    @staticmethod
    def expand_mask(mask: Tensor) -> Tensor:
        r"""
        This method expand (broadcast) the mask to support different mask shapes.
        Mask tensor for masking attention scores. 
        It must be of shape (batch_size, 1, sequence_length) 
        and contain 0s in positions where attention should be masked.

        :param mask: (torch.Tensor) mask tensor with ndim (dimensions) >= 2
        :return: mask (torch.Tensor) - expanded tensor mask with 4 ndim (dimensions) 
        """

        if mask.ndim < 2:
            raise AssertionError(
                f"mask must be at least 2-dimensional "
                f"with shape sequence_length x sequence_length, "
                f"but got mask with ndim={mask} and shape={mask.shape}"
            )

        if mask.ndim == 3:
            return mask.unsqueeze(1)
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)
        return mask

    def __compute_attention_scores(self,
                                   queries: Tensor,
                                   keys: Tensor,
                                   mask: Optional[Tensor] = None
                                   ) -> Tensor:
        r"""
        Computes attention scores as scaled dot product between queries and keys tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.

        Args:
            queries (torch.Tensor): queries features linear projection of shape [batch_size, num_heads, sequence_length, head_dim]. 
            keys (torch.Tensor): keys features linear projection with of shape [batch_size, num_heads, sequence_length, head_dim].
            mask (Optional[torch.Tensor]): Optional mask tensor for masking attention scores. 
                It should be of shape (batch_size, 1, sequence_length) and contain 0s in positions
                where attention should be masked.

        :return: attention_scores (torch.Tensor) - A tensor of shape 
            [batch_size, num_heads, sequence_length, sequence_length or numbers of keys or values tokens]
        """

        # dot product computing between queries and keys, where shapes Q @ K.T is:
        # [batch_size, num_heads, sequence_length, head_dim] @ [batch_size, num_heads, head_dim, sequence_length]
        # where sequence_length is equal numbers of keys tokens and can be different from queries sequence_length
        dot_product = queries @ keys.transpose(-2, -1)
        # dot product shape: [batch_size, num_heads, sequence_length, sequence_length]
        dot_product /= math.sqrt(queries.size(-1))  # scaling by square root of head_dim

        if mask is not None:
            mask = self.expand_mask(mask)
            # apply a mask to attention logits if one was specified
            # we replace dot_product values to infinity where mask values is 0 
            # since exp(inf) is 0 and masked values will not affect attention results
            dot_product = dot_product.masked_fill(mask == 0, torch.inf)

        attention_scores = F.softmax(dot_product, -1)  # applying softmax to get attention scores 
        return self.dropout(attention_scores)  # applying dropout to attention scores

    def _scaled_dot_product_attention(self,
                                      queries: Tensor,
                                      keys: Tensor,
                                      values: Tensor,
                                      mask: Tensor
                                      ) -> Tensor:
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.

        Args:
            queries (torch.Tensor): queries features linear projection of shape [batch_size, num_heads, sequence_length, head_dim]. 
            keys (torch.Tensor): keys features linear projection with of shape [batch_size, num_heads, sequence_length, head_dim].
            values (torch.Tensor): values features linear projection of shape [batch_size, num_heads, sequence_length, head_dim].
            mask (Optional[torch.Tensor]): Optional mask tensor for masking attention scores. 
                It should be of shape (batch_size, 1, sequence_length) and contain 0s in positions
                where attention should be masked.

        Outputs:
            - attention_scores (torch.Tensor) - a tensor of attention scores. Shape: [batch_size, num_heads, sequence_length, sequence_length]
            - attend (torch.Tensor) - values tensor where attention score was applied. Shape: [batch_size, num_heads, sequence_length, head_dim]

        Notes:
            `sequence_length` maybe not only the same number of tokens for queries, keys and values
                but also number of tokens specifically for keys and values and queries can have other sequence_length
        """

        # compute attention scores of shape [batch_size, num_heads, sequence_length, sequence_length]
        attention_scores = self.__compute_attention_scores(queries, keys, mask)
        attend = attention_scores @ values  # applying attention scores to values
        return attend, attention_scores

    def _separate_projections(self, linear_projection: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        r"""
        Separate (reshape) input linear projection of Q, K and V into heads 
        and split into chunks: 
            - queries, keys and values  (if self-attention only used)
            - keys and values  (if cross-attention only used)

        Notes:
            1. if not specified type of attention this method get a tensor with size [batch_size, sequence_length, embed_dim].
            It means that we get linear projection all heads only for queries or keys or values. 
            We reshape this linear projection by heads, permute into [batch_size, num_heads, sequence_length, dims] and return. 

            2. But if self-attention used we get linear projection of queries, keys and values
            of shape [batch_size, sequence_length, embed_dim * 3]
            Because of this we split tensor into chunks (Q, K, V) of size [batch_size, num_heads, sequence_length, head_dim] and then return.

            3. And if cross-attention used we get linear projection of queries, keys and values
            of shape [batch_size, sequence_length, embed_dim * 2]
            Because of this we split tensor into chunks (K, V) of size 
            [batch_size, num_heads, sequence_length, head_dim] and then return.

        :param linear_projection (torch.Tensor): all linear projections of 
            Q, K and V with shapes [batch_size, sequence_length, embed_dim * 3]
            or only one linear projection of Q or K or V 
            with shape [batch_size, sequence_length, embed_dim] for all heads
            or linear projection of KV of shape [batch_size, sequence_length, embed_dim * 2]

        Outputs: 
            - **linear_projection** - a linear projections of Q or K or V for all heads
                with shape [batch_size, num_heads, sequence_length, head_dim]

            OR

            - **queries**, **keys** and **values** - a separated linear projections of Q, K and V for all heads
                with shapes [batch_size, num_heads, sequence_length, head_dim] 

            OR

            - **keys** and **values** - a separated linear projections of K and V for all heads
                with shapes [batch_size, num_heads, sequence_length, head_dim] 
        """

        # input shape at this moment:
        #   [batch_size, sequence_length, embed_dim or embed_dim * 2 or embed_dim * 3]

        batch_size, seq_length, embed_dim = linear_projection.size()

        # reshape linear projection into one of variants below::
        #   1. [batch_size, sequence_length, num_heads, head_dim] - any forward (one linear projection Q or K or V for all heads)
        #   2. [batch_size, sequence_length, num_heads, head_dim * 3] - only self-attention (linear projections Q, K and V for all heads)
        #   3. [batch_size, sequence_length, num_heads, head_dim * 2] - only cross-attention (linear projections K and V for all heads)
        linear_projection = linear_projection.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim * (embed_dim // self.embed_dim)
        )

        # permute data into: 
        #   [batch_size, num_heads, sequence_length, dims]
        linear_projection = linear_projection.permute(0, 2, 1, 3)
        if (embed_dim // self.num_heads) == self.head_dim:
            # Q, K or V linear projection
            return linear_projection  # return linear_projection: [batch_size, num_heads, sequence_length, head_dim]
        elif (embed_dim // self.num_heads) == self.head_dim * 2:
            # K and V  linear projections
            return linear_projection.chunk(2, dim=-1)

        # split data into chunks: queries, keys, values with shape:
        #   [batch_size, num_heads, sequence_length, head_dim]
        return linear_projection.chunk(3, dim=-1)  # Q, K, V linear projections

    def _in_projection(self,
                       queries: Tensor,
                       keys: Tensor,
                       values: Tensor,
                       ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Do linear projection of Q, K and V for all heads into subspaces 
        using three separated weights matrices (nn.Linear layers) Wq, Wk and Wv for queries, keys and values projections.

        Note:
            kdim can maybe equal with vdim and embed_dim (embed_dim = kdim = vdim),
            but this method maybe used for any situations, include:
                - Q, K is V - K and V are same tensors (encoder-decoder case)
                - Q is not K is not V - is the different tensors with different dimensions (multi-modal case)
                - Q is K is V - is the same tensor (self-attention case)
                - etc.

        :param queries (torch.Tensor): Queries features of shape [batch_size, sequence_length, embed_dim].
        :param keys (torch.Tensor): Keys features with of shape [batch_size, sequence_length, kdim].
        :param values (torch.Tensor): Values features of shape [batch_size, sequence_length, vdim].


        :return: **queries**, **keys** and **values** - a linear projections for all heads
            with shape [batch_size, num_heads, sequence_length, head_dim].
        """

        # gets queries linear projection for all heads   
        queries = self.queries_projection(queries)  # shapes: [batch_size, sequence_length, embed_dim]
        queries = self._separate_projections(queries)  # shapes: [batch_size, num_heads, sequence_length, head_dim]

        # gets keys linear projection for all heads   
        keys = self.keys_projection(keys)
        keys = self._separate_projections(keys)  # and separate heads

        # gets values linear projection for all heads   
        values = self.values_projection(values)
        values = self._separate_projections(values)
        return queries, keys, values

    def _self_in_projection(self, qkv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Do linear projection of queries, keys, and values for all heads, 
        where queries, keys and values it the same tensor.

        Note:
            This method used only for self-attention case because
             make linear projection of Q, K, V only by one input (Q or K or V) 
             using one weights matrix (nn.Linear layer).

        :param qkv (torch.Tensor): queries or values or keys (any) features of shape [batch_size, sequence_length, embed_dim].

        :return: **queries**, **keys** and **values** - a linear projections for all heads.
            Shape: [batch_size, num_heads, sequence_length, head_dim]
        """

        # input shapes of Q, K and V at this stage: [batch_size, sequence_length, embed_dim]
        qkv = self.qkv_projection(qkv)  # make linear projection of Q, K and V for all heads into subspaces 
        # data shape at this stage [batch_size, sequence_length, embed_dim * 3]

        # split a linear projection into queries, keys and values tensors with size:
        #   [batch_size, num_heads, sequence_length, head_dim]
        return self._separate_projections(qkv)  # queries, keys, values  

    def _cross_in_projection(self, queries: Tensor, kv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Do linear projection of queries, keys, and values for all heads, 
        where keys and values it the same tensor.

        Note:
            This method used only for cross-attention case (encoder-decoder arthitectures)

        :param queries (torch.Tensor): Queries features of shape [batch_size, sequence_length, embed_dim].
        :param kv (torch.Tensor): keys or values (any) features of shape [batch_size, sequence_length, embed_dim].

        :return: **queries**, **keys** and **values** - a linear projections for all heads.
            Shape: [batch_size, num_heads, sequence_length, head_dim]
        """

        kv = self.kv_projection(kv)  # [batch_size, sequence_length, embed_dim * 2]
        queries = self.queries_projection(queries)  # [batch_size, sequence_length, embed_dim]
        keys, values = self._separate_projections(kv)  # [batch_size, num_heads, sequence_length, head_dim]
        queries = self._separate_projections(queries)  # [batch_size, num_heads, sequence_length, head_dim]
        return queries, keys, values

    @staticmethod
    def __check_qkv_shapes(
            queries: Tensor,
            keys: Tensor,
            values: Tensor
    ) -> None:
        # check shapes sanity         
        assert queries.ndim == keys.ndim == values.ndim, (
            f"queries, keys and values must have same ndim, "
            f"but got {queries.ndim=}, {keys.ndim=} and {values.ndim=}"
        )
        assert queries.ndim in [2, 3], (
            f"queries, keys and values must have at least 2 dims , "
            f"but got ndim={queries.ndim}"
        )
        assert keys.size(1) == values.size(1), (
            f"keys sequence_length {keys.size(1)} does not "
            f"match values sequence_length {values.size(1)}"
        )

    def forward(self,
                queries: Tensor,
                keys: Tensor,
                values: Tensor,
                mask: Optional[Tensor] = None,
                return_attention_probs: Optional[bool] = False
                ) -> Tensor:
        r"""
        Forward pass of the multi-head attention mechanism:

        .. math::
            \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

        where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

        Args:
            queries (torch.Tensor): Queries features of shape [batch_size, sequence_length, embed_dim].
            keys (torch.Tensor): Keys features with of shape [batch_size, sequence_length, embed_dim].
            values (torch.Tensor): Values features of shape [batch_size, sequence_length, embed_dim].
            mask (Optional[torch.Tensor]): Optional mask tensor for masking attention scores. 
                It should be of shape (batch_size, 1, sequence_length) and contain 0s in positions
                where attention should be masked.
            return_attention_probs (Optional[bool]): If True, return attention probabilities along with output.

        Notes:
            - Q, K and V - all maybe have both different and same features dimensions. 
            - K and V - must have same numbers of tokens (sequence_length), but it can be not equal numbers of Q tokens.

        Outputs:
            - attend (torch.Tensor): Output tensor of shape [batch_size, sequence_length, embed_dim].

            - attention_scores (torch.Tensor): tensor attention scores of shape [batch_size, num_heads, sequence_length, embed_dim].
        """

        # checking shapes sanity
        self.__check_qkv_shapes(queries, keys, values)
        # and expand batch_size dim if it needs
        if queries.ndim == 2:
            queries = queries.unsqueeze(0)
            values = values.unsqueeze(0)
            keys = keys.unsqueeze(0)

        # if during initialization attention_type specifed as 'self-attention' 
        # then will use 1 weights (Wqkv) for computing linear projection of QKV for all heads
        if self.qkv_projection is not None:
            assert queries is keys is values, "queries, keys and values must be same tensors for self-attention"
            queries, keys, values = self._self_in_projection(queries)  # only self-attention forward
        # or will use 2 separated weights (Wkv, Wq) 
        # for computing linear projection of Q and KV for all heads
        elif self.kv_projection is not None and self.queries_projection is not None:
            assert keys is values, "keys and values must be same tensors for cross-attention"
            queries, keys, values = self._cross_in_projection(queries, keys)  # only cross-attention forward
        else:
            # or will use 3 separated weights (Wq, Wk, Wv)
            # for computing each linear projection of Q, K and V for all heads
            queries, keys, values = self._in_projection(queries, keys, values)  # any attention forward

        # shapes for Q, K, V at this stage: [batch_size, num_heads, sequence_length, head_dim] 
        attend, attention_scores = self._scaled_dot_product_attention(
            queries, keys, values, mask
        )  # computing dot product attention

        # attend have shape [batch_size, num_heads, sequence_length, head_dim]
        # attention_scores have shape [batch_size, num_heads, sequence_length, sequence_length]
        attend = attend.permute(0, 2, 1, 3)  # [batch_size, sequence_length, num_heads, head_dim]
        attend = attend.reshape(
            attend.size(0), attend.size(1), self.embed_dim
        )  # [batch_size, sequence_length, num_heads * head_dim]
        # and make output linear projection for adding up results of all heads in one space representation 
        attend = self.output_projection(attend)  # [batch_size, sequence_length, embed_dim]

        if not return_attention_probs:
            return attend.squeeze()
        return attend.squeeze(), attention_scores.squeeze()
