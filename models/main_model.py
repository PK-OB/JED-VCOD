import torch
import torch.nn as nn
# import segmentation_models_pytorch as smp # (더 이상 사용하지 않음)
from .std_module import STDModule
from .dae_module import DAEModule # <--- Context-Gated DAEModule 임포트

class JED_VCOD_Fauna_Simplified(nn.Module):
    def __init__(self):
        super(JED_VCOD_Fauna_Simplified, self).__init__()

        # ▼▼▼ DAE를 수정된 DAEModule (Context-Gated U-Net)로 교체 ▼▼▼
        self.dae_features = [64, 128, 256, 512]
        self.dae = DAEModule(in_channels=3, features=self.dae_features)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # ▼▼▼ STD 모듈의 입력 채널을 DAEModule의 출력 채널에 맞게 설정 ▼▼▼
        # DAEModule.forward()는 [512, 256, 128, 64] 채널 순서 (저해상도->고해상도)로 반환.
        # STD 모듈은 이 순서를 그대로 기대합니다.
        std_in_channels = self.dae_features[::-1] # [512, 256, 128, 64]
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        self.std = STDModule(in_channels_list=std_in_channels, hidden_dim=128)

    def forward(self, video_clip):
        batch_size, seq_len, c, h, w = video_clip.shape
        # (B, T, C, H, W) -> (B*T, C, H, W)
        dae_input = video_clip.view(batch_size * seq_len, c, h, w)

        # ▼▼▼ 수정된 DAEModule의 forward() 호출 ▼▼▼
        # 반환 순서: [512, 256, 128, 64] 채널 (저해상도 -> 고해상도)
        multi_scale_features_flat = self.dae(dae_input)

        # STD 모듈이 DAEModule의 반환 순서([512, 256, 128, 64])를 그대로 기대하므로
        # reverse() 호출은 필요 없습니다. 이 줄을 확실히 제거합니다.
        # multi_scale_features_flat.reverse() # <--- ▼▼▼ 이 줄 제거 또는 주석 처리!!! ▼▼▼
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        multi_scale_features_seq_list = []
        # 이제 multi_scale_features_flat는 [feat_512, feat_256, feat_128, feat_64] 순서입니다.
        for features in multi_scale_features_flat:
            # (B*T, C_f, H_f, W_f) -> (B, T, C_f, H_f, W_f) -> (T, B, C_f, H_f, W_f)
            _, c_f, h_f, w_f = features.shape
            features_seq = features.view(batch_size, seq_len, c_f, h_f, w_f).permute(1, 0, 2, 3, 4)
            multi_scale_features_seq_list.append(features_seq)
        # multi_scale_features_seq_list도 [seq_512, seq_256, seq_128, seq_64] 순서가 됩니다.

        # STD 모듈은 std_in_channels=[512, 256, 128, 64]로 초기화되었고,
        # 입력 리스트도 동일한 순서이므로 채널 수가 일치하게 됩니다.
        predicted_masks_seq = self.std(multi_scale_features_seq_list, target_size=(h, w))

        return predicted_masks_seq