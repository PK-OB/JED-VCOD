# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/models/dae_module.py
# (ResNet 스킵 연결 로직이 완전히 수정된 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):
    """
    U-Net의 기본 구성 블록 (변경 없음)
    """
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

class AttentionGate(nn.Module):
    """
    Attention U-Net 게이트 (변경 없음)
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Gating signal, x: Skip connection
        g_prime = self.W_g(g)
        x_prime = self.W_x(x)
        
        # g_prime (from decoder)을 x_prime (from skip)의 크기로 업샘플링
        if g_prime.shape[2:] != x_prime.shape[2:]:
            g_prime = F.interpolate(g_prime, size=x_prime.shape[2:], mode='bilinear', align_corners=False)
            
        psi_input = self.relu(g_prime + x_prime)
        alpha = self.psi(psi_input)
        
        # 원본 스킵 연결(x)에 어텐션 맵(알파)을 곱함
        return x * alpha

class DAEModule(nn.Module):
    """
    SOTA 개선:
    1. 인코더 (ResNet-34)
    2. AttentionGate (버그 수정됨)
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # features는 하위 호환용
        super().__init__()
        
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        if in_channels != 3:
            self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.init_conv = resnet.conv1 # 1/2

        self.encoder0_0 = nn.Sequential(resnet.bn1, resnet.relu) # 1/2
        self.encoder0_1 = resnet.maxpool # 1/4
        self.encoder1 = resnet.layer1 # 64 ch, 1/4
        self.encoder2 = resnet.layer2 # 128 ch, 1/8
        self.encoder3 = resnet.layer3 # 256 ch, 1/16
        self.encoder4 = resnet.layer4 # 512 ch, 1/32
        
        self.bottleneck = ConvBlock(512, 1024) # 1/32
        
        # ▼▼▼ [버그 수정] 디코더와 스킵 연결 채널 재정의 ▼▼▼
        # 디코더 출력 채널
        self.decoder_channels = [512, 256, 128, 64] 
        # 스킵 연결 채널 (e3, e2, e1, x0_relu 순서)
        self.skip_channels = [256, 128, 64, 64] 
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        in_ch = 1024 # Bottleneck 출력
        
        for i in range(len(self.decoder_channels)):
            dec_ch = self.decoder_channels[i] # 512, 256, 128, 64
            skip_ch = self.skip_channels[i]   # 256, 128, 64, 64
            
            # Upconv layer
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, dec_ch, kernel_size=2, stride=2)
            )
            
            # Attention gate (Gating 'g': dec_ch, Skip 'x': skip_ch)
            self.attention_gates.append(
                AttentionGate(F_g=dec_ch, F_l=skip_ch, F_int=dec_ch // 2)
            )
            
            # Decoder conv block (Input: Upconv channel + Gated Skip channel)
            self.decoder_blocks.append(ConvBlock(dec_ch + skip_ch, dec_ch))
            
            in_ch = dec_ch # 다음 루프의 in_ch는 현재 루프의 out_ch

        # 이미지 복원 헤드
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(self.decoder_channels[-1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        # --- Encoder ---
        x0 = self.init_conv(x_in) # 64 ch, 1/2
        x0_relu = self.encoder0_0(x0) # 64 ch, 1/2 (Skip 3)
        
        e1_in = self.encoder0_1(x0_relu) # 1/4
        e1 = self.encoder1(e1_in) # 64 ch, 1/4 (Skip 2)
        
        e2 = self.encoder2(e1) # 128 ch, 1/8 (Skip 1)
        e3 = self.encoder3(e2) # 256 ch, 1/16 (Skip 0)
        e4 = self.encoder4(e3) # 512 ch, 1/32
        
        # ▼▼▼ [버그 수정] 스킵 연결 리스트 ▼▼▼
        # e4는 병목으로 가고, e3부터 스킵 연결 시작
        skip_connections = [e3, e2, e1, x0_relu] 
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        # --- Bottleneck ---
        b = self.bottleneck(e4) # 1024 ch, 1/32

        multi_scale_features = []
        x = b # 디코더 시작 (1024 ch, 1/32)

        # --- Decoder ---
        for i in range(len(self.decoder_channels)):
            # 1. 업샘플링
            x = self.upconvs[i](x) # i=0: (512 ch, 1/16)
            
            # 2. 스킵 연결 가져오기
            skip_connection = skip_connections[i] # i=0: e3 (256 ch, 1/16). 크기 일치!
            
            # 3. 어텐션 게이트
            gated_skip_connection = self.attention_gates[i](g=x, x=skip_connection)
            
            # 4. Concat
            concat_skip = torch.cat((gated_skip_connection, x), dim=1)
            
            # 5. 디코더 블록
            x = self.decoder_blocks[i](concat_skip) # i=0: out (512 ch, 1/16)
            
            multi_scale_features.append(x)

        # --- Reconstruction Head ---
        # x는 이제 마지막 디코더 출력 (64 ch, 1/2)
        reconstructed_image_half = self.reconstruction_head(x)
        
        # 원본 크기로 업샘플링
        reconstructed_image = F.interpolate(
            reconstructed_image_half, 
            size=x_in.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 특징 맵 리스트 (저해상도 -> 고해상도 순)와 복원 이미지 반환
        return multi_scale_features, reconstructed_image