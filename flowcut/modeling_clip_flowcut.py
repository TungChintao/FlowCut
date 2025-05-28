import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass,make_dataclass
from typing import Any, Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPConfig,CLIPVisionConfig
from transformers.utils import add_start_docstrings_to_model_forward,replace_return_docstrings
from functools import wraps
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from .utils_flowcut import compute_score, adaptive_prune_ratio, CLIP_VISION_INPUTS_DOCSTRING


def CLIPAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()
    # get query proj
    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    value_metric = value_states.clone()

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) 

    cls_value = value_states[:,:1]
    semantic_weight = torch.bmm(cls_value, value_states.transpose(1,2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    # apply the causal_attention_mask first
    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                f" {causal_attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)


    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    semantic_attn = nn.functional.softmax(semantic_weight, dim=-1).view(bsz, self.num_heads, src_len)

    if output_attentions:
        # this operation is a bit akward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)
    
    return attn_output, attn_weights_reshaped, semantic_attn.mean(1), value_metric.mean(1)


def CLIPEncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
    target: int = 577,
    rpl: int = 22,
    cumulative_scores: torch.Tensor = None,
    all_keep_indices: torch.Tensor = None,
) -> Tuple[torch.FloatTensor]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            `(config.encoder_attention_heads,)`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
    """
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, attn_weights, semantic_attn, v_metric = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states
    
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states 

    # ==================================== FlowCut ========================================== #
    if hidden_states.shape[1] > target:    

        cls_attention = attn_weights[:, :, 0, 1:].mean(dim=1)        # [B, N]
       
        scores = compute_score(cls_attention, semantic_attn[:,1:], v_metric[:,1:])  # [B, N]
        if cumulative_scores is None:
            cumulative_scores = scores.clone()
        else:
            cumulative_scores = 0.5*cumulative_scores+0.5*scores
       
        if not rpl%2:

            prune_num = adaptive_prune_ratio(cls_attention,rpl,target-1) if rpl else hidden_states.shape[1]
            target_num = max(hidden_states.shape[1]-prune_num, target)
  
            # keep 1 position for cls token
            _, keep_indices = torch.topk(cumulative_scores, target_num-1, dim=1)  # [B, target_num-1]
            
            keep_indices = keep_indices.sort(dim=1).values
            topk_states = torch.gather(hidden_states[:,1:,:], dim=1, index=keep_indices.unsqueeze(-1).expand(-1,-1,hidden_states.size(-1)))
            cumulative_scores = torch.gather(cumulative_scores, dim=1, index=keep_indices)
            cls_token = hidden_states[:,:1,:]

            all_keep_indices = torch.gather(all_keep_indices, dim=1, index=keep_indices)
            
            hidden_states = torch.cat([cls_token, topk_states], dim=1)

    # ==================================== FlowCut ========================================== #   

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs, cumulative_scores, all_keep_indices

def CLIPEncoder_forward(
    self,
    inputs_embeds,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutput]:

    # super().__init__()
    # self.config = config
    # self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    # self.gradient_checkpointing = False

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
   
    hidden_states = inputs_embeds
 
    all_keep_indices = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0],-1)
    cumulative_scores = None
    for idx, encoder_layer in enumerate(self.layers):
        remain_prune_layer = self.select_layer-idx-1
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs, cumulative_scores, all_keep_indices = self._gradient_checkpointing_func(
                encoder_layer.__call__,
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions,
                self.target,
                remain_prune_layer,
                cumulative_scores,
                all_keep_indices,
            )
        else:
            layer_outputs, cumulative_scores, all_keep_indices = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
                target = self.target,
                rpl = remain_prune_layer,
                cumulative_scores = cumulative_scores,
                all_keep_indices = all_keep_indices,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None) + (all_keep_indices,)
    
    return BaseModelOutput(
        last_hidden_state=hidden_states, 
        hidden_states=encoder_states, 
        attentions=all_attentions,
    ), all_keep_indices


@add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
def CLIPVisionTransformer_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")

    hidden_states = self.embeddings(pixel_values, **kwargs)
    hidden_states = self.pre_layrnorm(hidden_states)

    encoder_outputs, all_keep_indices = self.encoder(
        inputs_embeds=hidden_states,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    pooled_output = last_hidden_state[:, 0, :]
    pooled_output = self.post_layernorm(pooled_output)

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:] + (all_keep_indices, )

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    ), all_keep_indices


def apply_flowcut_config(model, target_tokens):

    model.vision_model.encoder.target = target_tokens
    model.vision_model.encoder.select_layer = len(model.vision_model.encoder.layers) - 2