# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/models/main_model.py

import torch
import torch.nn as nn
# import segmentation_models_pytorch as smp # (더 이상 사용하지 않음)
from .std_module import STDModule
from .dae_module import DAEModule # <--- 수정된 SOTA DAEModule 임포트

class JED_VCOD_Fauna_Simplified(nn.Module):
    def __init__(self):
        super(JED_VCOD_Fauna_Simplified, self).__init__()

        # ▼▼▼ SOTA DAE (ResNet-34 기반) ▼▼▼
        self.dae = DAEModule(in_channels=3)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # ▼▼▼ STD 모듈 입력 채널 설정 ▼▼▼
        # 수정된 DAEModule은 [512, 256, 128, 64] 채널 순서 (저해상도->고해상도)로 반환합니다.
        # (DAEModule의 decoder_channels 리스트와 일치)
        std_in_channels = [512, 256, 128, 64] 
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        self.std = STDModule(in_channels_list=std_in_channels, hidden_dim=128) # <--- 수정된 STDModule 임포트

    def forward(self, video_clip):
        batch_size, seq_len, c, h, w = video_clip.shape
        # (B, T, C, H, W) -> (B*T, C, H, W)
        dae_input = video_clip.view(batch_size * seq_len, c, h, w)

        # ▼▼▼ 수정된 DAEModule 호출 ▼▼▼
        # multi_scale_features_flat: [feat_512, feat_256, feat_128, feat_64] 순서
        multi_scale_features_flat, reconstructed_images_flat = self.dae(dae_input)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # DAE 출력 순서와 STD 입력 순서가 일치하므로 reverse()는 필요 없음.
        
        multi_scale_features_seq_list = []
        # multi_scale_features_flat는 [feat_512, feat_256, feat_128, feat_64] 순서
        for features in multi_scale_features_flat:
            # (B*T, C_f, H_f, W_f) -> (B, T, C_f, H_f, W_f) -> (T, B, C_f, H_f, W_f)
            _, c_f, h_f, w_f = features.shape
            features_seq = features.view(batch_size, seq_len, c_f, h_f, w_f).permute(1, 0, 2, 3, 4)
            multi_scale_features_seq_list.append(features_seq)
        # multi_scale_features_seq_list도 [seq_512, seq_256, seq_128, seq_64] 순서

        # STD 모듈은 std_in_channels=[512, 256, 128, 64]로 초기화되었고,
        # 입력 리스트도 동일한 순서이므로 채널 수가 일치하게 됩니다.
        predicted_masks_seq = self.std(multi_scale_features_seq_list, target_size=(h, w))

        # 분할 마스크와 복원된 이미지를 함께 반환
        return predicted_masks_seq, reconstructed_images_flat