import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
import transformers.models.llama.modeling_llama as llama_modeling

from transformers import modeling_utils

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

def hooked_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask,
    past_key_value = None,
    cache_position = None,
    output_attentions = False,
    ablated_heads = [],
    amplified_head = [],
    amplify_factor = 1.0,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = llama_modeling.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = llama_modeling.eager_attention_forward
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

    for head_index in amplified_head:
        if head_index < 0:
            head_index += attn_output.shape[2]  # Adjust for negative indexing
        if head_index >= attn_output.shape[2]:
            raise ValueError(f"Head index {head_index} is out of range for {self.num_heads} heads.")
        attn_output[:, :, head_index, :] *= amplify_factor + 1.0

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, head_outputs

def hooked_forward_for_llama3_decoder_layer(
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
    amplified_head = [],
    amplify_factor = 1.0,
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
        amplified_head=amplified_head,
        amplify_factor=amplify_factor,
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

class Llama3_injected(nn.Module):
    def __init__(
            self, 
            llama3_model : nn.Module, 
            auto_encoder : nn.Module, # NOTICE: only on the given token index
            injected_layer_num : int,
            hook = False,
            output_hidden_states = False,
            output_attentions = False,
            only_last_token_hidden_states = True, # if True, only return the last token hidden states, otherwise return all tokens hidden states (default: False)
            ablated_heads = {},
            amplified_head = {},
            amplify_factor = 1.0,
        ):
        super(Llama3_injected, self).__init__()
        self.llama3_model = llama3_model
        self.config = self.llama3_model.config
        try:
            self.auto_encoder = auto_encoder.to(self.llama3_model.device)
        except:
            self.auto_encoder = auto_encoder
        self.injected_layer_num = injected_layer_num
        if self.injected_layer_num > self.llama3_model.config.num_hidden_layers or self.injected_layer_num < 0:
            print("warning: injected_layer_num is out of range, will not be injected.")
        
        self.device = self.llama3_model.device
        self.hook = hook
        self.injected_attn = False

        if self.hook or any(ablated_heads.values()) or any(amplified_head.values()):
            self.make_hooks()

        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.only_last_token_hidden_states = only_last_token_hidden_states
        self.ablated_heads = ablated_heads
        self.amplified_head = amplified_head
        self.amplify_factor = amplify_factor
        for layer_index in range(len(self.llama3_model.model.layers)):
            if layer_index not in self.ablated_heads:
                self.ablated_heads[layer_index] = []
            if layer_index not in self.amplified_head:
                self.amplified_head[layer_index] = []
    
    def make_hooks(self):
        self.injected_attn = True
        for decoder_layer in self.llama3_model.model.layers[: self.llama3_model.model.config.num_hidden_layers]:
            decoder_layer.forward = types.MethodType(hooked_forward_for_llama3_decoder_layer, decoder_layer)
            decoder_layer.self_attn.forward = types.MethodType(hooked_attn_forward, decoder_layer.self_attn)

    def gradient_on(self, train_part = "both"):
        if train_part == 'none':
            return
        # Freeze all llama3_model parameters except lm_head
        for name, param in self.llama3_model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
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
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.llama3_model.model.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.llama3_model.model.gradient_checkpointing and self.llama3_model.model.training and use_cache:
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.llama3_model.model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.llama3_model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.llama3_model.model.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        layer_count = 0

        total_head_outputs = []
        encoding = None

        for decoder_layer in self.llama3_model.model.layers[: self.llama3_model.model.config.num_hidden_layers]:
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
                hidden_states = auto_encoder_res[0] * auto_encoder_mask + hidden_states * original_hidden_states_mask # 主要是为了不in place替换，同时保留injected_token_index之后的部分（如果label长度大于1）
                position_embeddings = (
                    position_embeddings[0][:, injected_token_index:, :], 
                    position_embeddings[1][:, injected_token_index:, :]
                )
                if return_encoded:
                    encoding = auto_encoder_res[1]

                causal_mask = create_causal_mask(
                    config=self.llama3_model.config,
                    input_embeds=hidden_states,
                    attention_mask=attention_mask,
                    cache_position=cache_position[:len(cache_position) - injected_token_index],
                    past_key_values=past_key_values,
                    position_ids=position_ids[:, :len(cache_position) - injected_token_index],
                )

            if output_hidden_states:
                if self.only_last_token_hidden_states:
                    all_hidden_states += (hidden_states[:, -1:, :].detach().to(torch.float).cpu().numpy(),)
                else:
                    all_hidden_states += (hidden_states.detach().to(torch.float).cpu().numpy(),)

            if self.injected_attn:
                layer_outputs, head_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    ablated_heads=self.ablated_heads[layer_count],
                    amplified_head=self.amplified_head[layer_count],
                    amplify_factor=self.amplify_factor,
                    **flash_attn_kwargs,
                )
                hidden_states = layer_outputs[0]
                if self.hook:
                    total_head_outputs.append(head_outputs.detach().to(torch.float).cpu().numpy())
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    ablated_heads=self.ablated_heads[layer_count],
                    amplified_head=self.amplified_head[layer_count],
                    amplify_factor=self.amplify_factor,
                    **flash_attn_kwargs,
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

        hidden_states = self.llama3_model.model.norm(hidden_states)

        # add hidden states from the last decoder layer
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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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
        logits = self.llama3_model.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.llama3_model.loss_function(logits=logits, labels=labels, vocab_size=self.llama3_model.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )