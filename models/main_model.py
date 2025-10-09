import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .std_module import STDModule

class JED_VCOD_Fauna_Simplified(nn.Module):
    def __init__(self):
        super(JED_VCOD_Fauna_Simplified, self).__init__()
        
        self.dae = smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        
        encoder_out_channels = self.dae.encoder.out_channels
        
        std_in_channels = list(encoder_out_channels[-4:])
        std_in_channels.reverse()
        
        self.std = STDModule(in_channels_list=std_in_channels, hidden_dim=128)

    def forward(self, video_clip):
        batch_size, seq_len, c, h, w = video_clip.shape
        dae_input = video_clip.view(batch_size * seq_len, c, h, w)
        
        multi_scale_features_flat = self.dae.encoder(dae_input)[-4:]
        multi_scale_features_flat.reverse()

        multi_scale_features_seq_list = []
        for features in multi_scale_features_flat:
            _, c_f, h_f, w_f = features.shape
            features_seq = features.view(batch_size, seq_len, c_f, h_f, w_f).permute(1, 0, 2, 3, 4)
            multi_scale_features_seq_list.append(features_seq)

        # STD 모듈을 호출할 때 원본 높이(h), 너비(w)를 함께 전달합니다.
        predicted_masks_seq = self.std(multi_scale_features_seq_list, target_size=(h, w))
        
        return predicted_masks_seq