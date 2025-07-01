from .modeling_qwen2_vl_flowcut import (
    VisionAttention_forward,
    VisionFlashAttention2_forward,
    VisionSdpaAttention_forward,
    Qwen2VLVisionBlock_forward,
    Qwen2VisionTransformerPretrainedModel_forward,
    Qwen2VLAttention_forward,
    Qwen2VLFlashAttention2_forward,
    Qwen2VLSdpaAttention_forward,
    Qwen2VLDecoderLayer_forward,
    Qwen2VLModel_forward,
    Qwen2VLForConditionalGeneration_forward
)

def flowcut(model, target_num=128):

    if target_num < 1:
        raise ValueError("target_tokens must ≥1")
    
    model.visual.target_num = target_num*2
    model.model.target_num = target_num
    
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        VisionAttention,
        VisionFlashAttention2,
        VisionSdpaAttention, 
        Qwen2VLVisionBlock, 
        Qwen2VisionTransformerPretrainedModel,
        Qwen2VLAttention,
        Qwen2VLFlashAttention2,
        Qwen2VLSdpaAttention,
        Qwen2VLDecoderLayer,
        Qwen2VLModel,
        Qwen2VLForConditionalGeneration
    )
    VisionAttention.forward = VisionAttention_forward
    VisionFlashAttention2.forward = VisionFlashAttention2_forward
    VisionSdpaAttention.forward = VisionSdpaAttention_forward

    Qwen2VLVisionBlock.forward = Qwen2VLVisionBlock_forward
    Qwen2VisionTransformerPretrainedModel.forward = Qwen2VisionTransformerPretrainedModel_forward
    Qwen2VLAttention.forward = Qwen2VLAttention_forward
    Qwen2VLFlashAttention2.forward = Qwen2VLFlashAttention2_forward
    Qwen2VLSdpaAttention.forward = Qwen2VLSdpaAttention_forward
    Qwen2VLDecoderLayer.forward = Qwen2VLDecoderLayer_forward
    Qwen2VLModel.forward = Qwen2VLModel_forward
    Qwen2VLForConditionalGeneration.forward = Qwen2VLForConditionalGeneration_forward

    return model

