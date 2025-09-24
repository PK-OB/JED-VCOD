import torch
import torch.nn as nn
from .dae_module import DAEModule
from .std_module import STDModule

class JED_VCOD_Fauna_Simplified(nn.Module):
    def __init__(self):
        super().__init__()
        self.dae_features = [64, 128, 256, 512]
        self.dae = DAEModule(in_channels=3, features=self.dae_features)
        self.std = STDModule(in_channels_list=self.dae_features[::-1], hidden_dim=128)

    def forward(self, video_clip):
        batch_size, seq_len, c, h, w = video_clip.shape
        dae_input = video_clip.view(batch_size * seq_len, c, h, w)
        multi_scale_features_flat = self.dae(dae_input)
        
        multi_scale_features_seq_list = []
        for features in multi_scale_features_flat:
            _, c_f, h_f, w_f = features.shape
            features_seq = features.view(batch_size, seq_len, c_f, h_f, w_f).permute(1, 0, 2, 3, 4)
            multi_scale_features_seq_list.append(features_seq)

        predicted_masks_seq = self.std(multi_scale_features_seq_list)
        return predicted_masks_seq