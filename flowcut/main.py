from .modeling_clip_flowcut import (
    CLIPAttention_forward, 
    CLIPEncoderLayer_forward, 
    CLIPEncoder_forward, 
    CLIPVisionTransformer_forward, 
    apply_flowcut_config,)
from .clip_encoder import CLIPVisionTower_FlowCut
from .modeling_llama_flowcut import (
    LlamaModel_forward, 
    LlamaDecoderLayer_forward, 
    LlamaSdpaAttention_forward,
    LlamaFlashAttention2_forward,)
from .llava_arch import (
    prepare_inputs_labels_for_multimodal_flowcut, 
    encode_images_flowcut, 
    encode_images_flowcut_multi, 
    restore_image_features_sorted)
from .llava_llama_flowcut import LlavaLlamaForCausalLM_forward,LlavaLlamaForCausalLM_generate


def flowcut(model, target_num=128):
   
    if target_num < 1:
        raise ValueError("target_tokens must ≥1")

    # llava-v1.5-7b: pad.    llava-v1.6-7b/llava-NeXT: anyres
    image_aspect_ratio = getattr(model.config, 'image_aspect_ratio', 'pad')
    # cls_patch: 1 cls token + 127 patch token
    # patch: 128 patch token
    
    # model.model.vision_tower.select_feature = 'cls_patch'
    select_feature = model.model.vision_tower.select_feature

    # target + 1 : keep one for cls token to calculate cls attn 
    #              and remove cls token at last
    if image_aspect_ratio == 'anyres':
        target_tokens = int(target_num/2) + 1 if select_feature == 'patch' else int(target_num/2)
    else:
        target_tokens = target_num + 1 if select_feature == 'patch' else target_num

    # config for visual token compression
    apply_flowcut_config(model.model.vision_tower.vision_tower, target_tokens=target_tokens)
    
    # replace CLIP component
    from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer, CLIPEncoder, CLIPVisionTransformer
    CLIPAttention.forward = CLIPAttention_forward
    CLIPEncoderLayer.forward = CLIPEncoderLayer_forward
    CLIPEncoder.forward = CLIPEncoder_forward

    CLIPVisionTransformer.forward = CLIPVisionTransformer_forward

    # replace Vision Tower
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_FlowCut.forward
    
    if image_aspect_ratio == 'anyres':
        model.model.target_num = target_num
        model.model.prune_layer = 1

        from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaSdpaAttention, LlamaFlashAttention2
        LlamaModel.forward = LlamaModel_forward
        LlamaDecoderLayer.forward = LlamaDecoderLayer_forward
        LlamaSdpaAttention.forward = LlamaSdpaAttention_forward
        LlamaFlashAttention2.forward = LlamaFlashAttention2_forward

        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        LlavaLlamaForCausalLM.forward = LlavaLlamaForCausalLM_forward
        LlavaLlamaForCausalLM.generate = LlavaLlamaForCausalLM_generate
    
    # replace multimodal processing
    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_flowcut
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_flowcut_multi = encode_images_flowcut_multi
        LlavaMetaForCausalLM.encode_images_flowcut = encode_images_flowcut

    return model