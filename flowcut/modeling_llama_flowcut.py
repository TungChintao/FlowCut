from typing import List, Optional, Tuple, Union, Any, Dict
import math
import torch
import torch.nn as nn
from transformers.utils import logging
from llava.constants import IGNORE_INDEX
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.llama.modeling_llama import (
    rotate_half, repeat_kv,
    apply_rotary_pos_emb,
)
from .utils_flowcut import compute_score


logger = logging.get_logger(__name__)


def LlamaFlashAttention2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    prune_kwargs: Optional[Dict[str, Any]] = {},
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    # ===================== new: relation_attn & semantic_attn ==============================
    if "prune" in prune_kwargs:
        # bsz, head, n, d
        _ = prune_kwargs.pop("prune")
        target_num = prune_kwargs.pop("target_num")
        visual_token_pos = prune_kwargs.pop("visual_token_pos")
        visual_token_num = prune_kwargs.pop("visual_token_num")
        last_instruct_pos = prune_kwargs.pop("last_instruct_pos")
        bsz = query_states.shape[0]
        keep_indices = []
        for b in range(bsz):
            if visual_token_num[b] <= target_num:
                continue

            start = visual_token_pos[b]
            end = start + visual_token_num[b]
            
            query_states = query_states.transpose(1,2)
            key_states = key_states.transpose(1,2)
            value_states = value_states.transpose(1,2)
            # vis_k: [head, vis_len, d]
            vis_k = key_states[b,:,start:end]
            # vis_v: [head, vis_len, d]
            vis_v = value_states[b,:,start:end]

            text_q = query_states[b,:,last_instruct_pos[b]-1:last_instruct_pos[b]]
            text_v = value_states[b,:,last_instruct_pos[b]-1:last_instruct_pos[b]]
            # text_q = query_states[b,:,-1:]
            # text_v = value_states[b,:,-1:]

            # re_attn: [head, 1, vis_len].mean(0) -> [1, vis_len]
            relation_weight = torch.bmm(text_q,vis_k.transpose(1,2))/math.sqrt(self.head_dim)
            relation_attn = nn.functional.softmax(relation_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)

            semantic_weight = torch.bmm(text_v, vis_v.transpose(1,2))/math.sqrt(self.head_dim)  # se_attn: [1,vis_len]
            semantic_attn = nn.functional.softmax(semantic_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)
       
            score = compute_score(relation_attn, semantic_attn, vis_v.mean(0))   # score: [1, vis_len]
     
            keep_index = torch.topk(score, target_num, dim=1).indices
            keep_indices.append(keep_index.sort().values)
        
        if keep_indices != []:
            keep_indices = torch.cat(keep_indices, dim=0)    # [b,target_num]
        
            return attn_output, attn_weights, past_key_value, keep_indices

    # ======================end: relation_attn & semantic_attn=================================

    return attn_output, attn_weights, past_key_value, None


# Adapted from LlamaAttention.forward
def LlamaSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    prune_kwargs: Optional[Dict[str, Any]] = {},
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
 
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        
    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )
 
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
  
    # ===================== new: relation_attn & semantic_attn ==============================
    if "prune" in prune_kwargs:
        # bsz, head, n, d
        _ = prune_kwargs.pop("prune")
        target_num = prune_kwargs.pop("target_num")
        visual_token_pos = prune_kwargs.pop("visual_token_pos")
        visual_token_num = prune_kwargs.pop("visual_token_num")
        last_instruct_pos = prune_kwargs.pop("last_instruct_pos")
        bsz = query_states.shape[0]
        keep_indices = []
        for b in range(bsz):
            if visual_token_num[b] <= target_num:
                continue

            start = visual_token_pos[b]
            end = start + visual_token_num[b]
            # vis_k: [head, vis_len, d]
            vis_k = key_states[b,:,start:end]
            # vis_v: [head, vis_len, d]
            vis_v = value_states[b,:,start:end]

            text_q = query_states[b,:,last_instruct_pos[b]-1:last_instruct_pos[b]]
            text_v = value_states[b,:,last_instruct_pos[b]-1:last_instruct_pos[b]]
            # text_q = query_states[b,:,-1:]
            # text_v = value_states[b,:,-1:]

            # re_attn: [head, 1, vis_len].mean(0) -> [1, vis_len]
            relation_weight = torch.bmm(text_q,vis_k.transpose(1,2))/math.sqrt(self.head_dim)
            relation_attn = nn.functional.softmax(relation_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)

            semantic_weight = torch.bmm(text_v, vis_v.transpose(1,2))/math.sqrt(self.head_dim)  # se_attn: [1,vis_len]
            semantic_attn = nn.functional.softmax(semantic_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)
       
            score = compute_score(relation_attn, semantic_attn, vis_v.mean(0))   # score: [1, vis_len]
     
            keep_index = torch.topk(score, target_num, dim=1).indices
            keep_indices.append(keep_index.sort().values)
        
        if keep_indices != []:
            keep_indices = torch.cat(keep_indices, dim=0)    # [b,target_num]
        
            return attn_output, None, past_key_value, keep_indices

    # ======================end: relation_attn & semantic_attn=================================

    return attn_output, None, past_key_value, None


def LlamaDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    prune_kwargs: Optional[Dict[str, Any]] = {},
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value, keep_indices = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        prune_kwargs=prune_kwargs,
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

    if use_cache:
        outputs += (present_key_value,)

    if keep_indices is not None:
        outputs += (keep_indices,) 

    return outputs



def LlamaModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    labels: Optional[torch.Tensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
  
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
   
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        past_key_values_length = past_key_values.get_usable_length(seq_length)
    
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)
   
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
   
    # Batch > 1 and seq_len num unequal(exist padding): attention_mask not None
    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # ========================================================================================
  
    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None    # new generated token's kv cache and pass to combinate the past_kv_cache

    # layer_outputs:tuple 
    # layer_outputs[0] = hidden_states layer[1] = attn_weight if output_attn else kv_cache
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        prune_kwargs = {}
        if layer_idx == self.prune_layer and hidden_states.shape[1] != 1:
            prune_kwargs["prune"] = True
            prune_kwargs["target_num"] = self.target_num
            prune_kwargs["visual_token_pos"] = self.visual_token_pos
            prune_kwargs["visual_token_num"] = self.visual_token_num
            prune_kwargs["last_instruct_pos"] = self.last_instruct_pos

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                prune_kwargs,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                prune_kwargs=prune_kwargs,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        
  
        # ==================== prune visual token at layer 1 ===============================
        if layer_idx == self.prune_layer:

            if hidden_states.shape[1] != 1:   # prefill stage or training
                _labels = labels
                _position_ids = position_ids
                _attention_mask = attention_mask

                if labels is None:
                    labels = torch.full((batch_size,hidden_states.shape[1]), IGNORE_INDEX, device=hidden_states.device)

                keep_indices = layer_outputs[-1] # [bsz, target_num]
                new_states_list = []
                new_label_list = []
                indices_num = 0
                for i in range(batch_size):
                    if self.visual_token_num[i] <= self.target_num:
                        new_states_list.append(hidden_states[i])
                        new_label_list.append(labels[i])
                        continue
                    start = self.visual_token_pos[i]
                    end = start+self.visual_token_num[i]
                    visual_states = hidden_states[i,start:end] # [vis_len, d]
                    topk_states = visual_states[keep_indices[indices_num]]

                    visual_label = labels[i, start:end]
                    topk_label = visual_label[keep_indices[indices_num]]

                    if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                        raise ValueError("Not support left padding now.")
                    else:
                        new_states = torch.cat([hidden_states[i,:start], topk_states, hidden_states[i,end:self.last_instruct_pos[i]]],dim=0)
                        label = torch.cat([labels[i,:start], topk_label, labels[i,end:self.last_instruct_pos[i]]])
                    #new_states = torch.cat([hidden_states[i,:start], topk_states, hidden_states[i,end:]],dim=0)
                    new_states_list.append(new_states)
                    indices_num += 1
                    new_label_list.append(label)

                tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
                if tokenizer_model_max_length is not None:
                    new_states_list = [x[:tokenizer_model_max_length] for x in new_states_list]
                
                max_len = max(x.shape[0] for x in new_states_list)

                new_states_padded = []

         
                position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=hidden_states.device)
                attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=hidden_states.device)
                new_labels = torch.full((batch_size,max_len), IGNORE_INDEX, device=hidden_states.device)

                for i, cur_new_states in enumerate(new_states_list): 
                    cur_len = cur_new_states.shape[0]
                    if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                        new_states_padded.append(torch.cat((
                            torch.zeros((max_len - cur_len, cur_new_states.shape[1]), dtype=cur_new_states.dtype, device=cur_new_states.device),
                            cur_new_states
                        ), dim=0))
                        if cur_len > 0:
                            new_labels[i, -cur_len:] = new_label_list[i]
                            position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                            attention_mask[i, -cur_len:] = True
                    else:
                        new_states_padded.append(torch.cat((
                            cur_new_states,
                            torch.zeros((max_len - cur_len, cur_new_states.shape[1]), dtype=cur_new_states.dtype, device=cur_new_states.device)
                        ), dim=0))
                        if cur_len > 0:
                            new_labels[i, :cur_len] = new_label_list[i]
                            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                            attention_mask[i, :cur_len] = True

                hidden_states = torch.stack(new_states_padded, dim=0).to(new_states_list[0].dtype)

                if _labels is None:
                    new_labels = None
                if _position_ids is None:
                    position_ids = None
                if _attention_mask is None:
                    attention_mask = None

            else:
                position_ids -= torch.clamp(torch.as_tensor(self.visual_token_num, device=position_ids.device, dtype=position_ids.dtype) - self.target_num, min=0)
                new_labels = labels
   
        # ============================== prune visual token at layer 1 ===============================================
    
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    
    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    ),new_labels



    
    

    
