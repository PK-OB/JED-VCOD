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

class DAEModule(nn.Module):
    # ▼▼▼ 수정된 부분 ▼▼▼
    # features 매개변수는 항상 외부에서 값을 전달받으므로 '='를 삭제합니다.
    def __init__(self, in_channels=3, features=[]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconvs = nn.ModuleList()

        # 인코더
        for feature in features:
            self.encoder_blocks.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # 디코더
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(ConvBlock(feature * 2, feature))

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

    def forward(self, x):
        skip_connections = []
        # 인코더 경로
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        multi_scale_features = []
        # 디코더 경로 및 다중 스케일 특징 맵 반환
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip_connection = skip_connections[i]

            # 크기가 맞지 않는 경우를 대비해 리사이즈 (홀수 해상도 등)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_blocks[i](concat_skip)
            multi_scale_features.append(x)

        return multi_scale_features