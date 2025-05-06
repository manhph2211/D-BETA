import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import MultiHeadAttention, LayerNorm


class SwitchGate(nn.Module):
    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor, use_aux_loss=False):
        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        capacity = int(self.capacity_factor * x.size(0))
        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)
        mask = torch.zeros_like(gate_scores).scatter_(1, top_k_indices, 1)
        masked_gate_scores = gate_scores * mask
        denominators = (masked_gate_scores.sum(0, keepdim=True) + self.epsilon)
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            load = gate_scores.sum(0)
            importance = gate_scores.sum(1)
            loss = ((load - importance) ** 2).mean()
            return gate_scores, loss

        return gate_scores, None


class SwitchMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, dim)
                )
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(dim, num_experts, capacity_factor)

    def forward(self, x: torch.Tensor):
        gate_scores, loss = self.gate(x, use_aux_loss=self.use_aux_loss)
        expert_outputs = [expert(x) for expert in self.experts]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )
        return moe_output, loss


class SwitchTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: float = 768,
        n_heads: float = 12,
        ffn_dim: float = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layer_norm_first: bool = False,
        num_experts: int = 4,  # Number of experts for SwitchMoE
        capacity_factor: float = 1.0,  # Capacity factor for SwitchMoE
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        def gelu(x: torch.Tensor) -> torch.Tensor:
            return F.gelu(x.float()).type_as(x)
        self.activation_fn = gelu
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            n_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.switch_moe = SwitchMoE(
            embed_dim,
            ffn_dim,
            num_experts,
            capacity_factor,
            use_aux_loss=False,  # Enabling the auxiliary loss
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            
            # Pass through the SwitchMoE layer
            x, moe_loss = self.switch_moe(x)
            x = self.dropout2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x

        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            
            # Pass through the SwitchMoE layer
            x, moe_loss = self.switch_moe(x)
            x = self.dropout2(x)

            layer_result = x

            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result, moe_loss)
