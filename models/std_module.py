# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/models/std_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Attention Block (변경 없음)"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvLSTMCell(nn.Module):
    """ConvLSTMCell (변경 없음)"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class STDModule(nn.Module):
    """
    SOTA 개선:
    1. 특징 융합을 덧셈(+=)에서 Concat + 1x1 Conv 방식으로 변경
    2. ConvLSTM의 연산 해상도를 1/8로 고정하여 효율성 확보
    """
    def __init__(self, in_channels_list, hidden_dim=128):
        super(STDModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_fusion_convs = nn.ModuleList()
        
        total_fused_channels = 0 # <-- 1. SOTA 개선 (Concat을 위함)
        
        # in_channels_list는 [512, 256, 128, 64] 순서
        for in_channels in in_channels_list:
            # 1x1 Conv로 각 스케일의 채널을 hidden_dim으로 통일
            self.feature_fusion_convs.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            )
            total_fused_channels += hidden_dim # <-- 1. SOTA 개선
            
        # ▼▼▼ 1. SOTA 개선: Concat + 1x1 Conv 융합 레이어 ▼▼▼
        # 덧셈 대신 Concat을 사용하므로, (채널 수 * 스케일 수) -> hidden_dim으로 줄이는 Conv 추가
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_fused_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        self.attention = SEBlock(hidden_dim)
        
        self.conv_lstm = ConvLSTMCell(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=(3, 3))

        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

    def forward(self, multi_scale_features_seq_list, target_size):
        # multi_scale_features_seq_list는 [seq_512, seq_256, seq_128, seq_64] 순서
        
        # ▼▼▼ 2. SOTA 개선: 융합 기준 해상도 변경 (1/8 크기) ▼▼▼
        # DAE (ResNet)의 1/8 해상도 특징 맵(list[1])의 크기를 기준으로 ConvLSTM을 돌립니다.
        # (e.g., 224x224 입력 -> 28x28에서 LSTM 연산)
        if len(multi_scale_features_seq_list) > 1:
            lstm_size = multi_scale_features_seq_list[1].shape[3:] # (H/8, W/8)
        else:
            lstm_size = multi_scale_features_seq_list[0].shape[3:] # (단일 스케일 입력 대비)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        seq_len = multi_scale_features_seq_list[0].shape[0]
        batch_size = multi_scale_features_seq_list[0].shape[1]
        device = multi_scale_features_seq_list[0].device
        
        h, c = self.conv_lstm.init_hidden(batch_size, lstm_size, device)

        outputs = []
        for t in range(seq_len):
            
            # ▼▼▼ 1. SOTA 개선: Concat 융합 ▼▼▼
            features_resized_list = []
            
            for i, features_seq in enumerate(multi_scale_features_seq_list):
                # 각 스케일의 특징 맵을 1x1 Conv로 채널 통일
                feature_t = self.feature_fusion_convs[i](features_seq[t])
                # 통일된 융합 기준 크기(lstm_size)로 리사이즈
                feature_t_resized = F.interpolate(feature_t, size=lstm_size, mode='bilinear', align_corners=False)
                features_resized_list.append(feature_t_resized)
            
            # 리스트에 담긴 텐서들을 채널(dim=1) 기준으로 Concat
            fused_feature_cat = torch.cat(features_resized_list, dim=1)
            # 1x1 Conv로 융합하여 채널 수를 다시 hidden_dim으로 줄임
            fused_feature = self.fusion_conv(fused_feature_cat)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            
            fused_feature = self.attention(fused_feature) # (SEBlock은 그대로 사용)

            h, c = self.conv_lstm(fused_feature, (h, c))
            pred_mask = self.seg_head(h)
            
            # 최종 출력은 원본 크기(target_size)로 업샘플링
            pred_mask_upsampled = F.interpolate(pred_mask, size=target_size, mode='bilinear', align_corners=False)
            outputs.append(pred_mask_upsampled)
        
        return torch.stack(outputs, dim=0).permute(1, 0, 2, 3, 4)