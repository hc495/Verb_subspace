import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers import modeling_utils

import types



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def hooked_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask,
    past_key_value = None,
    cache_position = None,
    output_attentions = False,
    ablated_heads = [],
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager" and (not output_attentions):
        attention_interface = modeling_utils.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    head_outputs = attn_output.clone()

    for head_index in ablated_heads:
        if head_index < 0:
            head_index += attn_output.shape[2]  # Adjust for negative indexing
        if head_index >= attn_output.shape[2]:
            raise ValueError(f"Head index {head_index} is out of range for {self.num_heads} heads.")
        attn_output[:, :, head_index, :] = 0.0

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, head_outputs

def hooked_forward_for_qwen2_decoder_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    cache_position = None,
    position_embeddings = None,  # necessary, but kept here for BC
    ablated_heads = [],
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, head_outputs = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        ablated_heads=ablated_heads,
        **kwargs,
    )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)
    return outputs, head_outputs

class Qwen2_injected(nn.Module):
    def __init__(
            self, 
            qwen2_model : nn.Module, 
            auto_encoder : nn.Module, # NOTICE: only on the given token index
            injected_layer_num : int,
            hook = False,
            output_hidden_states = False,
            output_attentions = False,
            only_last_token_hidden_states = True, # if True, only return the last token hidden states, otherwise return all tokens hidden states (default: False)
            ablated_heads = {},
        ):
        super(Qwen2_injected, self).__init__()
        self.qwen2_model = qwen2_model
        self.config = self.qwen2_model.config
        self.auto_encoder = auto_encoder.to(self.qwen2_model.device)
        self.injected_layer_num = injected_layer_num
        if self.injected_layer_num > self.qwen2_model.config.num_hidden_layers or self.injected_layer_num < 0:
            print("warning: injected_layer_num is out of range, will not be injected.")

        self.device = self.qwen2_model.device
        self.hook = hook
        self.injected_attn = False

        if self.hook or any(ablated_heads.values()):
            self.make_hooks()

        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.only_last_token_hidden_states = only_last_token_hidden_states
        self.ablated_heads = ablated_heads
        for layer_index in range(len(self.qwen2_model.model.layers)):
            if layer_index not in self.ablated_heads:
                self.ablated_heads[layer_index] = []

    def make_hooks(self):
        self.injected_attn = True
        # Register modified member functions
        for decoder_layer in self.qwen2_model.model.layers[: self.qwen2_model.model.config.num_hidden_layers]:
            decoder_layer.forward = types.MethodType(hooked_forward_for_qwen2_decoder_layer, decoder_layer)
            decoder_layer.self_attn.forward = types.MethodType(hooked_attn_forward, decoder_layer.self_attn)

    def gradient_on(self, train_part = "both"):
        if train_part == 'none':
            return
        # Freeze all qwen2_model parameters except lm_head
        for name, param in self.qwen2_model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if self.qwen2_model.config.tie_word_embeddings:
            self.qwen2_model.model.embed_tokens.weight.requires_grad = True
        # Enable gradients for auto_encoder
        for param in self.auto_encoder.parameters():
            if train_part == "both" or train_part == "auto_encoder":
                param.requires_grad = True
            else:
                param.requires_grad = False
        if train_part == "encoder":
            for param in self.auto_encoder.encoder.parameters():
                param.requires_grad = True
        if train_part == "decoder":
            for param in self.auto_encoder.decoder.parameters():
                param.requires_grad = True  

    def new_inner_inference(
        self,
        input_ids = None,
        injected_token_index = None, # int
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        cache_position = None,
        return_encoded = False,
        forced_encoded = None,
        task_vector = None,
        task_vector_layer = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.qwen2_model.model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.qwen2_model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.qwen2_model.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.qwen2_model.model.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        layer_count = 0
        encoding = None

        total_head_outputs = []

        for decoder_layer in self.qwen2_model.model.layers[: self.qwen2_model.config.num_hidden_layers]:
            if task_vector_layer is not None and task_vector is not None and layer_count == task_vector_layer:
                for batch_index in range(hidden_states.shape[0]):
                    hidden_states[batch_index, -1] += task_vector
            if layer_count == self.injected_layer_num:
                # inject the autoencoder
                # cut
                hidden_states = hidden_states[:, injected_token_index:, :].detach()
                auto_encoder_mask = torch.zeros_like(hidden_states)
                auto_encoder_mask[:, 0, :] = 1
                auto_encoder_mask = auto_encoder_mask.to(hidden_states.device)
                auto_encoder_mask.requires_grad_(False)
                original_hidden_states_mask = torch.ones_like(hidden_states)
                original_hidden_states_mask[:, 0, :] = 0
                original_hidden_states_mask = original_hidden_states_mask.to(hidden_states.device)
                original_hidden_states_mask.requires_grad_(False)
                auto_encoder_res = self.auto_encoder(hidden_states, return_encoded, forced_encoded)
                hidden_states = auto_encoder_res[0] * auto_encoder_mask + hidden_states * original_hidden_states_mask
                position_embeddings = (
                    position_embeddings[0][:, injected_token_index:, :], 
                    position_embeddings[1][:, injected_token_index:, :]
                )
                if return_encoded:
                    encoding = auto_encoder_res[1]
                
                mask_kwargs = {
                    "config": self.qwen2_model.config,
                    "input_embeds": hidden_states,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position[:len(cache_position) - injected_token_index],
                    "past_key_values": past_key_values,
                    "position_ids": position_ids[:, :len(cache_position) - injected_token_index],
                }
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                # The sliding window alternating layers are not always activated depending on the config
                if self.qwen2_model.model.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

            if output_hidden_states:
                if self.only_last_token_hidden_states:
                    all_hidden_states += (hidden_states[:, -1:, :].detach().to(torch.float).cpu().numpy(),)
                else:
                    all_hidden_states += (hidden_states.detach().to(torch.float).cpu().numpy(),)
            
            if self.injected_attn:
                layer_outputs, head_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    ablated_heads=self.ablated_heads[layer_count],
                    **kwargs,
                )
                hidden_states = layer_outputs[0]
                if self.hook:
                    total_head_outputs.append(head_outputs.detach().to(torch.float).cpu().numpy())
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    ablated_heads=self.ablated_heads[layer_count],
                    **kwargs,
                )
                hidden_states = layer_outputs
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            layer_count += 1

        if task_vector_layer is not None and task_vector is not None and layer_count == task_vector_layer:
                for batch_index in range(hidden_states.shape[0]):
                    hidden_states[batch_index, -1] += task_vector
                    
        if layer_count == self.injected_layer_num:
            hidden_states = hidden_states[:, injected_token_index:, :].detach()
            auto_encoder_mask = torch.zeros_like(hidden_states)
            auto_encoder_mask[:, 0, :] = 1
            auto_encoder_mask = auto_encoder_mask.to(hidden_states.device)
            auto_encoder_mask.requires_grad_(False)
            original_hidden_states_mask = torch.ones_like(hidden_states)
            original_hidden_states_mask[:, 0, :] = 0
            original_hidden_states_mask = original_hidden_states_mask.to(hidden_states.device)
            original_hidden_states_mask.requires_grad_(False)
            hidden_states = self.auto_encoder(hidden_states) * auto_encoder_mask + hidden_states * original_hidden_states_mask
            position_embeddings = (
                position_embeddings[0][:, injected_token_index:, :], 
                position_embeddings[1][:, injected_token_index:, :]
            )

        hidden_states = self.qwen2_model.model.norm(hidden_states)

        if output_hidden_states:
            if self.only_last_token_hidden_states:
                all_hidden_states += (hidden_states[:, -1:, :].detach().to(torch.float).cpu().numpy(),)
            else:
                all_hidden_states += (hidden_states.detach().to(torch.float).cpu().numpy(),)

        if (not self.hook) and (not return_encoded):
            hidden_states_dic = all_hidden_states
        else:
            hidden_states_dic = {'hidden_states' : all_hidden_states}
            if return_encoded:
                hidden_states_dic['encoding'] = encoding.detach().to(torch.float).cpu().numpy()
            if self.hook:
                hidden_states_dic['head_outputs'] = total_head_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=hidden_states_dic,
            attentions=all_self_attns,
        )

    def forward(
        self,
        input_ids = None,
        injected_token_index = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        cache_position = None,
        logits_to_keep = 0,
        return_encoded = False,
        forced_encoded = None,
        task_vector = None,
        task_vector_layer = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.new_inner_inference(
            input_ids=input_ids,
            injected_token_index=injected_token_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_encoded=return_encoded,
            forced_encoded=forced_encoded,
            task_vector=task_vector,
            task_vector_layer=task_vector_layer,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.qwen2_model.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.qwen2_model.loss_function(logits=logits, labels=labels, vocab_size=self.qwen2_model.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )