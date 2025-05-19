# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

## Thanks to the fairseq-signals project for these modules. Please refer https://github.com/Jwoo5/fairseq-signals/tree/master/fairseq_signals/modules for the original code.

from .fp32_group_norm import Fp32GroupNorm
from .grad_multiply import GradMultiply
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .layer_norm import Fp32LayerNorm, LayerNorm
from .multi_head_attention import MultiHeadAttention
from .same_pad import SamePad
from .switch_transformer_encoder_layer import SwitchTransformerEncoderLayer
from .transformer_encoder_layer import TransformerEncoderLayer
from .transformer_encoder import TransformerEncoder
from .transpose_last import TransposeLast
from .conv_feature_extraction import ConvFeatureExtraction, TransposedConvFeatureExtraction
from .conv_positional_encoding import ConvPositionalEncoding
from .utils import compute_mask_indices

__all__ = [
    "ConvFeatureExtraction",
    "ConvPositionalEncoding",
    "Fp32GroupNorm",
    "GradMultiply",
    "GumbelVectorQuantizer",
    "Fp32LayerNorm",
    "LayerNorm",
    "MultiHeadAttention",
    "SamePad",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "TransposeLast",
    "TransposedConvFeatureExtraction",
    "SwitchTransformerEncoderLayer",
    "compute_mask_indices"
]