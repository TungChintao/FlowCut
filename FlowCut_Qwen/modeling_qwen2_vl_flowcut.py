from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_rotary_pos_emb_vision, 
    Qwen2VLCausalLMOutputWithPast,
    rotate_half, repeat_kv,
    apply_multimodal_rotary_pos_emb,
)

from .utils_flowcut import compute_score_vit, compute_score_llm, adaptive_prune_ratio

def VisionAttention_forward(
    self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    attention_mask = torch.full(
        [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
    attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)

    # # ==================================== FlowCut ========================================== #
    global_q = q.mean(1,keepdim=True)
    global_v = v.mean(1,keepdim=True)

    # attn: [head, 1, N].mean(0) -> [1, N]
    attn_weight =  torch.bmm(global_q, k.transpose(1,2)) / math.sqrt(q.shape[-1])
    attn = F.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype).mean(0)

    score = compute_score_vit(attn, v.mean(0))

    # # ==================================== FlowCut ========================================== #  
    return attn_output, score, attn

def VisionFlashAttention2_forward(
    self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)

    # # ==================================== FlowCut ========================================== #
    global_q = q.mean(1,keepdim=True)
    global_v = v.mean(1,keepdim=True)

    # attn: [head, 1, N].mean(0) -> [1, N]
    attn_weight =  torch.bmm(global_q, k.transpose(1,2)) / math.sqrt(q.shape[-1])
    attn = F.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype).mean(0)

    score = compute_score_vit(attn, v.mean(0))

    # # ==================================== FlowCut ========================================== #  

    return attn_output, score, attn


def VisionSdpaAttention_forward(
    self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

    q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
    k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)

    # # ==================================== FlowCut ========================================== #
    # # q,k,v: head,N,D
    # # [head, 1, D]
    #global_q,_,global_v = self.qkv(hidden_states.mean(0,keepdim=True)).reshape(1, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

    global_q = q.mean(1,keepdim=True)
    global_v = v.mean(1,keepdim=True)

    # attn: [head, 1, N].mean(0) -> [1, N]
    attn_weight =  torch.bmm(global_q, k.transpose(1,2)) / math.sqrt(q.shape[-1])
    attn = F.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype).mean(0)

    score = compute_score_vit(attn, v.mean(0))

    # # ==================================== FlowCut ========================================== #  

    return attn_output, score, attn


def Qwen2VLVisionBlock_forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
  
    attn_output, score, relation_attn = self.attn(
        self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
    )
    hidden_states = hidden_states + attn_output
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    
    return hidden_states, score, relation_attn


def Qwen2VisionTransformerPretrainedModel_forward(
    self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
) -> torch.Tensor:
   
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # ================================= FlowCut Setting ===================================
    cumulative_scores = None
    total_layer = len(self.blocks)
    target = self.target_num*4  # keep target_num*4 / prune to target num in LLM 2 layer
    # ================================= FlowCut Setting ===================================

    for idx, blk in enumerate(self.blocks):
        hidden_states, score, relation_attn = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

    # ==================================== FlowCut ========================================== #
        cur_seq_len = hidden_states.shape[0]
        if cur_seq_len > target:
            remain_layer = total_layer-idx-1
            if cumulative_scores is None:
                cumulative_scores = score.squeeze(0)
            else:
                cumulative_scores = 0.5*cumulative_scores + 0.5*score.squeeze(0)
        
            if not remain_layer % 2:
                prune_num = adaptive_prune_ratio(relation_attn,remain_layer,target) if remain_layer else cur_seq_len
                target_num = max(cur_seq_len-prune_num, target)
                keep_indices = torch.topk(cumulative_scores, target_num, dim=0).indices
                keep_indices = keep_indices.sort(dim=0).values.to(hidden_states.device)
                cumulative_scores = cumulative_scores[keep_indices]
                hidden_states = hidden_states[keep_indices]
                rotary_pos_emb = rotary_pos_emb[keep_indices]
    
    return self.merger(hidden_states)


def Qwen2VLAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Fix precision issues in Qwen2-VL float16 inference
    # Replace inf values with zeros in attention weights to prevent NaN propagation
    if query_states.dtype == torch.float16:
        attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    
    # ===================== new: relation_attn & semantic_attn ==============================
    if "prune" in kwargs:
        # bsz, head, n, d
        visual_token_pos = kwargs.pop("visual_token_pos")
        visual_token_num = kwargs.pop("visual_token_num")
        target_num = kwargs.pop("target_num")
        bsz = query_states.shape[0]

        if visual_token_num > target_num:
    
            start = visual_token_pos
            end = start + visual_token_num
            # vis_k: [head, vis_len, d]
            vis_k = key_states[:,:,start:end].squeeze(0)
            # vis_v: [head, vis_len, d]
            vis_v = value_states[:,:,start:end].squeeze(0)

            text_q = query_states[:,:,-1:].squeeze(0)
            text_v = value_states[:,:,-1:].squeeze(0)
         
            # re_attn: [head, 1, vis_len].mean(0) -> [1, vis_len]
            relation_weight = torch.bmm(text_q,vis_k.transpose(1,2))/math.sqrt(self.head_dim)
            relation_attn = nn.functional.softmax(relation_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)
            
            semantic_weight = torch.bmm(text_v, vis_v.transpose(1,2))/math.sqrt(self.head_dim)  # se_attn: [1,vis_len]
            semantic_attn = nn.functional.softmax(semantic_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)
       
            score = compute_score_llm(relation_attn, semantic_attn, vis_v.mean(0))   # score: [1, vis_len]
     
            keep_index = torch.topk(score, target_num, dim=1).indices
            keep_indices = keep_index.sort().values.squeeze(0)
        
            return attn_output, attn_weights, past_key_value, keep_indices

    # ======================end: relation_attn & semantic_attn=================================

    return attn_output, attn_weights, past_key_value, None

def Qwen2VLFlashAttention2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
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

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    
    # ===================== new: relation_attn & semantic_attn ==============================
    if "prune" in kwargs:
        # bsz, head, n, d
        visual_token_pos = kwargs.pop("visual_token_pos")
        visual_token_num = kwargs.pop("visual_token_num")
        target_num = kwargs.pop("target_num")
        bsz = query_states.shape[0]

        if visual_token_num > target_num:
    
            start = visual_token_pos
            end = start + visual_token_num
            # vis_k: [head, vis_len, d]
            vis_k = key_states[:,:,start:end].squeeze(0)
            # vis_v: [head, vis_len, d]
            vis_v = value_states[:,:,start:end].squeeze(0)

            text_q = query_states[:,:,-1:].squeeze(0)
            text_v = value_states[:,:,-1:].squeeze(0)
         
            # re_attn: [head, 1, vis_len].mean(0) -> [1, vis_len]
            relation_weight = torch.bmm(text_q,vis_k.transpose(1,2))/math.sqrt(self.head_dim)
            relation_attn = nn.functional.softmax(relation_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)
            
            semantic_weight = torch.bmm(text_v, vis_v.transpose(1,2))/math.sqrt(self.head_dim)  # se_attn: [1,vis_len]
            semantic_attn = nn.functional.softmax(semantic_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)
       
            score = compute_score_llm(relation_attn, semantic_attn, vis_v.mean(0))   # score: [1, vis_len]
     
            keep_index = torch.topk(score, target_num, dim=1).indices
            keep_indices = keep_index.sort().values.squeeze(0)
        
            return attn_output, attn_weights, past_key_value, keep_indices

    # ======================end: relation_attn & semantic_attn=================================

    return attn_output, attn_weights, past_key_value, None


def Qwen2VLSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2VLModel is using Qwen2VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    # ===================== new: relation_attn & semantic_attn ==============================
    if "prune" in kwargs:
        # bsz, head, n, d
        visual_token_pos = kwargs.pop("visual_token_pos")
        visual_token_num = kwargs.pop("visual_token_num")
        target_num = kwargs.pop("target_num")
        bsz = query_states.shape[0]

        if visual_token_num > target_num:
    
            start = visual_token_pos
            end = start + visual_token_num
            # vis_k: [head, vis_len, d]
            vis_k = key_states[:,:,start:end].squeeze(0)
            # vis_v: [head, vis_len, d]
            vis_v = value_states[:,:,start:end].squeeze(0)

            text_q = query_states[:,:,-1:].squeeze(0)
            text_v = value_states[:,:,-1:].squeeze(0)
         
            # re_attn: [head, 1, vis_len].mean(0) -> [1, vis_len]
            relation_weight = torch.bmm(text_q,vis_k.transpose(1,2))/math.sqrt(self.head_dim)
            relation_attn = nn.functional.softmax(relation_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)
            
            semantic_weight = torch.bmm(text_v, vis_v.transpose(1,2))/math.sqrt(self.head_dim)  # se_attn: [1,vis_len]
            semantic_attn = nn.functional.softmax(semantic_weight, dim=-1, dtype=torch.float32).to(query_states.dtype).mean(0)
       
            score = compute_score_llm(relation_attn, semantic_attn, vis_v.mean(0))   # score: [1, vis_len]
     
            keep_index = torch.topk(score, target_num, dim=1).indices
            keep_indices = keep_index.sort().values.squeeze(0)
        
            return attn_output, None, past_key_value, keep_indices

    # ======================end: relation_attn & semantic_attn=================================

    return attn_output, None, past_key_value, None


def Qwen2VLDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """

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
        cache_position=cache_position,
        position_embeddings=position_embeddings,
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


def Qwen2VLModel_forward(
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
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for layer_idx, decoder_layer in enumerate(self.layers):
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        kwargs_param = {}

        if layer_idx == 1 and hidden_states.shape[1] != 1 and self.visual_token_num > self.target_num:
            kwargs_param["prune"] = True
            kwargs_param["target_num"] = self.target_num
            kwargs_param["visual_token_pos"] = self.visual_token_pos
            kwargs_param["visual_token_num"] = self.visual_token_num

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                **kwargs_param,
            )
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
                **kwargs_param,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # ================= prune visual token at layer 1 (second layer) =====================
        
        if layer_idx == 1 and hidden_states.shape[1] != 1 and self.visual_token_num > self.target_num:
            keep_indices = layer_outputs[-1]
            start = self.visual_token_pos
            end = start+self.visual_token_num
            visual_states = hidden_states[:,start:end]
            visual_position = position_ids[:,:,start:end]
            topk_states = visual_states[:,keep_indices]
            topk_position = visual_position[:,:,keep_indices]
            cache_position = cache_position[keep_indices] 

            causal_mask = self._update_causal_mask(None, hidden_states, cache_position, None, output_attentions)
            hidden_states = torch.cat([hidden_states[:,:start], topk_states, hidden_states[:,end:]],dim=1)
            position_ids = torch.cat([position_ids[:,:,:start], topk_position, position_ids[:,:,end:]],dim=2)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # if layer_idx == 1 and hidden_states.shape[1] != 1:
        #     token_num = self.target_num if self.visual_token_num > self.target_num else self.visual_token_num
        #     update_counter(token_num, "add_new")
      
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )



def Qwen2VLForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
           
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            #image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            # indices = torch.randperm(image_embeds.shape[0])
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_indices = torch.where(input_ids[0] == self.config.image_token_id)[0]
            first,last = image_indices[0],image_indices[-1]
            self.model.visual_token_pos = first
            self.model.visual_token_num = image_embeds.shape[0]
            inputs_embeds = torch.cat([
                inputs_embeds[:,:first], 
                image_embeds.unsqueeze(0), 
                inputs_embeds[:,last+1:]
            ], dim=1)
   
            position_ids = torch.cat([
                position_ids[:,:,:first],
                position_ids[:,:,first:first+image_embeds.shape[0]],
                position_ids[:,:,last+1:]
            ],dim=-1)
            
            #inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            self.model.visual_token_num = -1
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
    
    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )