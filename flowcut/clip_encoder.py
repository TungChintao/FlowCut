import torch
import torch.nn as nn

class CLIPVisionTower_FlowCut(nn.Module):

    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            image_features = []
            keep_indices = []
            for image in images:
                image_forward_out, keep_index = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
                keep_indices.append(keep_index)
        else:
            outputs, keep_indices = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            hidden_states = self.feature_select(outputs)
            # keep_indices = outputs.all_keep_indices

        return hidden_states.to(images.dtype), keep_indices

