# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/models/dae_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# ▼▼▼ ContextGate 수정: 출력을 1채널로 변경 ▼▼▼
class ContextGate(nn.Module):
    """
    병목 특징 맵을 사용하여 문맥 기반 게이트(1채널 어텐션 맵)를 생성합니다.
    """
    def __init__(self, in_channels):
        super().__init__()
        # 1x1 Conv로 채널 수를 1로 줄이고 Sigmoid로 (0~1) 사이의 게이트 맵 생성
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True), # <-- 채널 수를 1로 변경
            nn.Sigmoid()
        )

    def forward(self, x):
        # x는 게이트를 생성할 병목 특징 맵
        # 1채널 게이트를 생성하여 반환
        return self.gate_conv(x)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

class DAEModule(nn.Module):
    """
    Context-Gated 스킵 커넥션을 사용하는 U-Net 기반 DAE 모듈.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # features 기본값 설정
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconvs = nn.ModuleList()

        # 인코더
        current_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(ConvBlock(current_channels, feature))
            current_channels = feature

        # 디코더
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # 디코더 블록의 입력 채널은 (업샘플링된 채널 + 게이트 적용된 스킵 커넥션 채널)
            self.decoder_blocks.append(ConvBlock(feature * 2, feature))

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # ContextGate 초기화 (입력 채널은 bottleneck과 동일)
        self.context_gate_generator = ContextGate(in_channels=features[-1] * 2)
        
        # ▼▼▼ 수정된 부분: 이미지 복원 헤드 추가 ▼▼▼
        # 디코더의 첫 번째 블록(가장 고해상도) 출력 채널(features[0])을 입력으로 받음
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0] // 2, 3, kernel_size=1),
            nn.Sigmoid() # 출력을 [0, 1] 범위로 매핑 (원본 주간 이미지와 비교 위함)
        )
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    def forward(self, x):
        skip_connections = []
        # 인코더 경로: 각 블록 통과 후 스킵 커넥션 저장
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 병목 구간 통과 (가장 깊은 문맥 정보 추출)
        x_bottleneck = self.bottleneck(x)

        # 병목 특징으로 1채널 문맥 게이트 생성
        context_gate = self.context_gate_generator(x_bottleneck) # Shape: [B*T, 1, H/32, W/32]

        # 스킵 커넥션 순서 뒤집기 (디코더에서 사용하기 위함)
        skip_connections = skip_connections[::-1]

        multi_scale_features = [] # STD 모듈에 전달할 특징 맵 리스트

        x = x_bottleneck # 디코더 시작점

        # 디코더 경로
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x) # 업샘플링
            skip_connection = skip_connections[i] # 해당 레벨의 스킵 커넥션 가져오기

            # ▼▼▼ 게이트 적용 로직 (RuntimeError 해결) ▼▼▼

            # (1) 1채널 문맥 게이트(context_gate)를 현재 스킵 커넥션의 크기로 리사이즈
            current_gate = F.interpolate(context_gate, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            # current_gate Shape: [B*T, 1, H_skip, W_skip]

            # (2) 게이트 적용: skip_connection(다중 채널) * current_gate(1 채널)
            # PyTorch의 브로드캐스팅(broadcasting) 덕분에 채널 수가 달라도 곱셈 가능
            gated_skip_connection = skip_connection * current_gate

            # (3) 업샘플링된 특징(x)과 게이트 적용된 스킵 커넥션(gated_skip_connection)을 concat
            if x.shape != gated_skip_connection.shape:
                x = F.interpolate(x, size=gated_skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((gated_skip_connection, x), dim=1)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            # 디코더 블록 통과
            x = self.decoder_blocks[i](concat_skip)

            # STD 모듈에 전달할 특징 맵 저장 (고해상도 -> 저해상도 순서로 저장됨)
            multi_scale_features.append(x)

        # ▼▼▼ 수정된 부분: 복원된 이미지 생성 및 반환 ▼▼▼
        # x는 이제 가장 고해상도 디코더 특징 맵 (예: 64채널)
        reconstructed_image = self.reconstruction_head(x)
        
        # multi_scale_features 리스트는 [Decoder_out_1(64ch), Decoder_out_2(128ch), ...] 순서
        # 특징 맵 리스트와 복원된 이미지를 함께 반환
        return multi_scale_features, reconstructed_image
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲