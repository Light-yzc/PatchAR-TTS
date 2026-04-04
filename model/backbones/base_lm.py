import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

try:
    from torch.nn.attention.bias import causal_lower_right
except ImportError:
    causal_lower_right = None

try:
    from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import MoeCausalLMOutputWithPast
except ImportError:
    from dataclasses import dataclass

    class PretrainedConfig:
        model_type = "custom"

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class PreTrainedModel(nn.Module):
        config_class = None

        def __init__(self, config):
            super().__init__()
            self.config = config

    class GenerationMixin:
        pass

    @dataclass
    class MoeCausalLMOutputWithPast:
        loss: torch.Tensor | None = None
        aux_loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        past_key_values: list | None = None
        hidden_states: torch.Tensor | None = None


class MiniMindConfig(PretrainedConfig):
    """
    A compact causal Transformer config with later-layer MoE support.

    Layout:
    - early layers: dense
    - middle / upper layers: MoE FFN
    - final layers: dense tail
    """

    model_type = "minimind"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        vocab_size=6400,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=None,
        intermediate_size=None,
        hidden_act="silu",
        max_position_embeddings=32768,
        dropout=0.0,
        flash_attn=True,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        inference_rope_scaling=False,
        use_moe=True,
        dense_prefix_layers=4,
        dense_tail_layers=2,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=None,
        norm_topk_prob=True,
        router_aux_loss_coef=5e-4,
        router_use_input_norm=True,
        router_logits_clip=8.0,
        router_softmax_fp32=True,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.flash_attn = flash_attn

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or math.ceil(hidden_size * math.pi / 64) * 64

        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if inference_rope_scaling
            else None
        )

        self.use_moe = use_moe
        self.dense_prefix_layers = dense_prefix_layers
        self.dense_tail_layers = dense_tail_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size or self.intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_use_input_norm = router_use_input_norm
        self.router_logits_clip = router_logits_clip
        self.router_softmax_fp32 = router_softmax_fp32

        self._validate()

    def _validate(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.dense_prefix_layers < 0 or self.dense_tail_layers < 0:
            raise ValueError("dense_prefix_layers and dense_tail_layers must be >= 0")
        if self.dense_prefix_layers + self.dense_tail_layers > self.num_hidden_layers:
            raise ValueError("dense_prefix_layers + dense_tail_layers cannot exceed num_hidden_layers")
        if self.num_key_value_heads <= 0 or self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.use_moe:
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive")
            if self.num_experts_per_tok <= 0 or self.num_experts_per_tok > self.num_experts:
                raise ValueError("num_experts_per_tok must be in [1, num_experts]")

    @property
    def num_moe_layers(self):
        if not self.use_moe:
            return 0
        return self.num_hidden_layers - self.dense_prefix_layers - self.dense_tail_layers


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def precompute_freqs_cis(dim, end, rope_base=1e6, rope_scaling=None):
    """
    Precompute rotary cos/sin tables.

    The optional YaRN scaling branch is kept because it was already in your code,
    but the implementation is written more directly.
    """

    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float()[: dim // 2] / dim))
    attention_factor = 1.0

    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attention_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:
            def inv_dim(beta):
                return (dim * math.log(orig_max / (beta * 2 * math.pi))) / (2 * math.log(rope_base))

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.arange(dim // 2, device=freqs.device).float()
            ramp = torch.clamp((ramp - low) / max(high - low, 1e-3), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    positions = torch.arange(end, device=freqs.device)
    angles = torch.outer(positions, freqs).float()
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attention_factor
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attention_factor
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q, k


def repeat_kv(x, n_rep):
    batch_size, seq_len, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
    return x.reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_repeats = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.use_flash = hasattr(F, "scaled_dot_product_attention") and config.flash_attn

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _causal_mask(self, query_len, key_len, past_len, device):
        return torch.tril(
            torch.ones(query_len, key_len, device=device, dtype=torch.bool),
            diagonal=past_len,
        )

    def _build_sdpa_mask(self, attention_mask, query_len, key_len, past_len, device):
        causal_mask = self._causal_mask(query_len, key_len, past_len, device).view(1, 1, query_len, key_len)
        if attention_mask is None:
            return causal_mask
        key_mask = attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)
        return causal_mask & key_mask

    def forward(
        self,
        x,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        batch_size, query_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, query_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, query_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, query_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.shape[1]
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_key_value = (k, v) if use_cache else None

        k = repeat_kv(k, self.num_repeats)
        v = repeat_kv(v, self.num_repeats)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        key_len = k.shape[-2]
        attn_output = None

        if self.use_flash:
            try:
                if attention_mask is None and past_key_value is None:
                    attn_output = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True,
                    )
                else:
                    if attention_mask is None and causal_lower_right is not None:
                        sdpa_mask = causal_lower_right(query_len, key_len)
                    else:
                        sdpa_mask = self._build_sdpa_mask(
                            attention_mask=attention_mask,
                            query_len=query_len,
                            key_len=key_len,
                            past_len=past_len,
                            device=q.device,
                        )
                    attn_output = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=sdpa_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=False,
                    )
            except RuntimeError:
                attn_output = None

        if attn_output is None:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            causal_mask = self._causal_mask(query_len, key_len, past_len, scores.device)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            if attention_mask is not None:
                key_mask = attention_mask[:, None, None, :].to(torch.bool)
                scores = scores.masked_fill(~key_mask, float("-inf"))

            attn_probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
            attn_probs = self.attn_dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, present_key_value


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size=None):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    """
    Simple top-k MoE FFN.

    Only the FFN is sparse. Attention remains dense.
    That is the usual "safer first version" for an MoE LLM.
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_logits_clip = config.router_logits_clip
        self.router_softmax_fp32 = config.router_softmax_fp32

        self.router_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.router_use_input_norm else None
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                FeedForward(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        # Flatten tokens across batch/time so routing happens per token.
        # x:      [B, T, H]
        # flat_x: [B*T, H]
        flat_x = x.reshape(-1, hidden_dim)
        router_input = flat_x
        if self.router_input_norm is not None:
            router_input = self.router_input_norm(router_input)

        # Router scores over experts for every token.
        # router_logits / router_probs: [num_tokens, num_experts]
        router_logits = self.router(router_input)
        if self.router_logits_clip is not None:
            router_logits = router_logits.clamp(min=-self.router_logits_clip, max=self.router_logits_clip)
        if self.router_softmax_fp32:
            router_probs = F.softmax(router_logits.float(), dim=-1)
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        # For each token, keep only top-k experts.
        # topk_weights / topk_experts: [num_tokens, top_k]
        topk_weights, topk_experts = torch.topk(
            router_probs,
            k=self.num_experts_per_tok,
            dim=-1,
            sorted=False,
        )
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # Accumulate the weighted expert outputs back into the flattened token grid.
        # output: [num_tokens, H]
        output = torch.zeros_like(flat_x)

        for expert_idx, expert in enumerate(self.experts):
            # token_ids: which flattened tokens selected this expert
            # route_ids: whether it was selected as top-1 / top-2 / ...
            token_ids, route_ids = (topk_experts == expert_idx).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue

            # expert_input:  [num_routed_tokens, H]
            # expert_output: [num_routed_tokens, H]
            expert_input = flat_x.index_select(0, token_ids)
            expert_output = expert(expert_input)
            # expert_weight: [num_routed_tokens, 1]
            expert_weight = topk_weights[token_ids, route_ids].unsqueeze(-1).to(expert_output.dtype)
            output.index_add_(0, token_ids, expert_output * expert_weight)

        # Count every selected top-k route, not just the top-1 expert.
        # selected_expert_mask: [num_tokens, top_k, num_experts]
        selected_expert_mask = F.one_hot(topk_experts, num_classes=self.num_experts).to(router_probs.dtype)
        # Average selection frequency per expert across tokens.
        # tokens_per_expert: [top_k, num_experts]
        tokens_per_expert = selected_expert_mask.mean(dim=0)
        # Average router preference for each expert before top-k truncation.
        # avg_router_prob_per_expert: [num_experts]
        avg_router_prob_per_expert = router_probs.mean(dim=0)

        # Load-balancing auxiliary loss:
        # - gets larger if routing mass and actual token assignments both collapse
        #   onto a small subset of experts
        # - gets smaller when tokens and router probability are spread more evenly
        #   across experts
        #
        # This auxiliary term is returned to the LM, summed across MoE layers in
        # MiniMindModel.forward(), and then added into the final training loss in
        # LMTTSModel.forward() as `weighted_moe_aux_loss`.
        aux_loss = (
            self.num_experts
            * torch.sum(tokens_per_expert * avg_router_prob_per_expert.unsqueeze(0))
            * self.router_aux_loss_coef
        )
        aux_loss = aux_loss.to(flat_x.dtype)

        output = output.view(batch_size, seq_len, hidden_dim)
        return output, aux_loss


class MiniMindBlock(nn.Module):
    def __init__(self, config: MiniMindConfig, use_moe_ffn=False):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MoEFeedForward(config) if use_moe_ffn else FeedForward(config)
        self.use_moe_ffn = use_moe_ffn

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        attn_input = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            attn_input,
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        ffn_input = self.post_attention_layernorm(hidden_states)
        if self.use_moe_ffn:
            ffn_output, aux_loss = self.mlp(ffn_input)
        else:
            ffn_output = self.mlp(ffn_input)
            aux_loss = hidden_states.new_zeros(())

        hidden_states = residual + ffn_output
        return hidden_states, present_key_value, aux_loss


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [
                MiniMindBlock(config, use_moe_ffn=self._layer_uses_moe(layer_idx))
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def enable_gradient_checkpointing(self) -> None:
        """Recompute transformer layers on backward to reduce activation memory."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def _layer_uses_moe(self, layer_idx):
        if not self.config.use_moe:
            return False
        moe_start = self.config.dense_prefix_layers
        moe_end = self.config.num_hidden_layers - self.config.dense_tail_layers
        return moe_start <= layer_idx < moe_end

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        inputs_embeds=None,
        **kwargs,
    ):
        del kwargs

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        batch_size, seq_len, _ = hidden_states.shape

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        past_len = 0
        if past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[1]

        if attention_mask is None:
            total_len = past_len + seq_len
            attention_mask = torch.ones(batch_size, total_len, device=hidden_states.device, dtype=torch.long)

        position_embeddings = (
            self.freqs_cos[past_len : past_len + seq_len],
            self.freqs_sin[past_len : past_len + seq_len],
        )

        hidden_states = self.dropout(hidden_states)

        presents = []
        total_aux_loss = hidden_states.new_zeros(())

        if self.gradient_checkpointing and self.training and use_cache:
            # KV caching and activation checkpointing solve opposite problems.
            # Training uses checkpointing; inference uses KV cache.
            use_cache = False

        for layer, layer_past in zip(self.layers, past_key_values):
            if self.gradient_checkpointing and self.training:
                if layer_past is not None:
                    raise ValueError("gradient checkpointing does not support past_key_values during training")

                hidden_states, present, layer_aux_loss = checkpoint(
                    layer,
                    hidden_states,
                    position_embeddings,
                    None,
                    False,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states, present, layer_aux_loss = layer(
                    hidden_states,
                    position_embeddings,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
            presents.append(present)
            total_aux_loss = total_aux_loss + layer_aux_loss

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents, total_aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config=None):
        config = config or MiniMindConfig()
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        inputs_embeds=None,
        **kwargs,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_for_logits = hidden_states[:, -logits_to_keep:, :]
        else:
            hidden_for_logits = hidden_states

        logits = self.lm_head(hidden_for_logits)
        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        **kwargs,
    ):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(num_return_sequences, 1)

        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0

            outputs = self.forward(
                input_ids=input_ids[:, past_len:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

            if attention_mask is not None:
                extra_mask = attention_mask.new_ones(attention_mask.shape[0], 1)
                attention_mask = torch.cat([attention_mask, extra_mask], dim=-1)

            logits = outputs.logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for batch_idx in range(input_ids.shape[0]):
                    seen_tokens = torch.unique(input_ids[batch_idx])
                    logits[batch_idx, seen_tokens] /= repetition_penalty

            if top_k > 0:
                threshold = torch.topk(logits, top_k)[0][..., -1, None]
                logits[logits < threshold] = -float("inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = 0
                logits[remove_mask.scatter(1, sorted_indices, remove_mask)] = -float("inf")

            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                eos_fill = next_token.new_full((next_token.shape[0], 1), eos_token_id)
                next_token = torch.where(finished.unsqueeze(-1), eos_fill, next_token)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None

            if streamer is not None:
                streamer.put(next_token.cpu())

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer is not None:
            streamer.end()

        if kwargs.get("return_kv"):
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids


BaseLMConfig = MiniMindConfig
BaseLM = MiniMindModel
