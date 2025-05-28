import torch

CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*, defaults `False`):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

def compute_score(cls_attn, semantic_attn, value):

    relation_score = cls_attn/cls_attn.sum(dim=-1, keepdim=True)
    semantic_score = semantic_attn/semantic_attn.sum(dim=-1, keepdim=True)
   
    final_score = (relation_score + semantic_score) * torch.norm(value, p=1, dim=-1)#/value.shape[-1]

    return final_score


def adaptive_prune_ratio(attn,rpl,target_num):
    
    N = attn.shape[-1]

    prob = attn/attn.sum(dim=-1,keepdim=True)
    entropy = -torch.sum(prob * torch.log(prob), dim=-1)
    entropy_max = torch.log(torch.tensor(N)) 
    entropy_ratio = entropy/entropy_max
   
    r = (N-target_num)/rpl**0.5 * (1-entropy_ratio**2)
    if r.shape[0] > 1: r = r.max(0).values
    
    return int(r.round())