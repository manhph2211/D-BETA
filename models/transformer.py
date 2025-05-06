
import logging
import os
os.environ['CURL_CA_BUNDLE'] = ''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.modules import (
    GradMultiply,
    LayerNorm,
    ConvFeatureExtraction,
    ConvPositionalEncoding,
    TransformerEncoderLayer,
    MultiHeadAttention,
    LayerNorm,
    compute_mask_indices
)

from .base import PretrainingModel
logger = logging.getLogger(__name__)


def init_bert_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiHeadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embed_dim = args.encoder_embed_dim
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer( 
                    embed_dim=self.embed_dim,
                    ffn_dim=args.encoder_ffn_embed_dim,
                    n_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    layer_norm_first=args.layer_norm_first,
                ) for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embed_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(
        self,
        x,
        padding_mask=None,
        attn_mask=None,
    ):
        x = self.extract_features(x, padding_mask, attn_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(
        self,
        x,
        padding_mask=None,
        attn_mask=None
    ):
        if padding_mask is not None:
            x[padding_mask] = 0

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    self_attn_mask=attn_mask,
                    need_weights=False
                )
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0,1)

        return x


class TransformerModel(PretrainingModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        if cfg.apply_mask:
            self.mask_emb = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
            )

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.num_updates = 0

        self.encoder = TransformerEncoder(cfg)

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap = self.no_mask_channel_overlap,
                min_space = self.mask_channel_min_space
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward(
        self,
        x,
        padding_mask=None,
        **kwargs
    ):
        raise NotImplementedError()

    def extract_features(self, source, padding_mask):
        raise NotImplementedError()


class ECGTransformerModel(TransformerModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtraction(
            conv_layers=feature_enc_layers,
            in_d=cfg.in_d,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.feature_grad_mult = cfg.feature_grad_mult
        self.conv_pos = ConvPositionalEncoding(cfg)
        self.layer_norm = LayerNorm(self.embed)
    
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2])

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        **kwargs
    ):
        x, padding_mask = self.get_embeddings(source, padding_mask)
        x = self.get_output(x, padding_mask)
        return {"x": x, "padding_mask": padding_mask}

    def get_embeddings(self, source, padding_mask):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = features.transpose(1,2)
        features = self.layer_norm(features)

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            if input_lengths.dim() > 1:
                for input_len in input_lengths:
                    assert (input_len == input_len[0]).all()
                input_lengths = input_lengths[:,0]
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype = features.dtype, device = features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device = padding_mask.device),
                    output_lengths - 1
                )
            ] = 1
            padding_mask[torch.where(output_lengths == 0)] = 0
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        
        features = self.dropout_input(features)

        x = features
        x_conv = self.conv_pos(x, channel_first=False)
        x = x + x_conv

        return x, padding_mask

    def get_output(self, x, padding_mask=None):
        x = self.encoder(x, padding_mask=padding_mask)
        return x

    def extract_features(self, source, padding_mask):
        res = self.forward(source, padding_mask=padding_mask)
        return res

    def get_logits(self, net_output, normalize=False, aggregate=False):
        logits = net_output["x"]

        if net_output["padding_mask"] is not None and net_output["padding_mask"].any():
            logits[net_output["padding_mask"]] = 0
        
        if aggregate:
            logits = torch.div(logits.sum(dim=1), (logits != 0).sum(dim=1))
        
        return logits

    def get_targets(self, net_output):
        raise NotImplementedError()
