import torch
import torch.nn as nn
from models.dae_module import DAEModule
from models.std_module import STDModule

class JED_VCOD_Fauna_Simplified(nn.Module):
    def __init__(self):
        super().__init__()
        # ▼▼▼ 수정된 부분 ▼▼▼
        # DAE 모듈(U-Net)의 각 인코더 레벨에서 사용할 특징 맵의 채널 수를 정의합니다.
        self.dae_features = [64, 128, 256, 512] 
        
        self.dae = DAEModule(in_channels=3, features=self.dae_features)
        # DAE 디코더의 출력 채널 순서(고해상도->저해상도 특징맵)를 STD 모듈의 입력으로 전달합니다.
        self.std = STDModule(in_channels_list=self.dae_features[::-1], hidden_dim=128)

    def forward(self, video_clip):
        # 입력: video_clip (B, T, C, H, W)
        batch_size, seq_len, c, h, w = video_clip.shape
        
        # DAE 모듈은 프레임 단위로 처리하기 위해 텐서 형태 변경 (B*T, C, H, W)
        dae_input = video_clip.view(batch_size * seq_len, c, h, w)
        multi_scale_features_flat = self.dae(dae_input)
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        # STD 모듈 입력을 위해 리스트를 초기화합니다.
        multi_scale_features_seq_list = []
        for features in multi_scale_features_flat:
            _, c_f, h_f, w_f = features.shape
            # (B*T, C, H, W) -> (B, T, C, H, W) -> (T, B, C, H, W)
            features_seq = features.view(batch_size, seq_len, c_f, h_f, w_f).permute(1, 0, 2, 3, 4)
            multi_scale_features_seq_list.append(features_seq)

        # STD 모듈에 강화된 특징 맵 시퀀스 리스트 전달
        predicted_masks_seq = self.std(multi_scale_features_seq_list)
        
        # 최종 출력: (B, T, 1, H, W)
        return predicted_masks_seq