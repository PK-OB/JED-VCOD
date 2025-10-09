import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Attention Block"""
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
    def __init__(self, in_channels_list, hidden_dim=128):
        super(STDModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_fusion_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.feature_fusion_convs.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))

        self.attention = SEBlock(hidden_dim)
        
        self.conv_lstm = ConvLSTMCell(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=(3, 3))

        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

    def forward(self, multi_scale_features_seq_list, target_size):
        seq_len = multi_scale_features_seq_list[0].shape[0]
        batch_size = multi_scale_features_seq_list[0].shape[1]
        device = multi_scale_features_seq_list[0].device
        
        lstm_size = multi_scale_features_seq_list[0].shape[3:]
        h, c = self.conv_lstm.init_hidden(batch_size, lstm_size, device)

        outputs = []
        for t in range(seq_len):
            fused_feature = torch.zeros(batch_size, self.hidden_dim, *lstm_size, device=device)
            for i, features_seq in enumerate(multi_scale_features_seq_list):
                feature_t = self.feature_fusion_convs[i](features_seq[t])
                feature_t_resized = F.interpolate(feature_t, size=lstm_size, mode='bilinear', align_corners=False)
                fused_feature += feature_t_resized
            
            fused_feature = self.attention(fused_feature)

            h, c = self.conv_lstm(fused_feature, (h, c))
            pred_mask = self.seg_head(h)
            
            pred_mask_upsampled = F.interpolate(pred_mask, size=target_size, mode='bilinear', align_corners=False)
            outputs.append(pred_mask_upsampled)
        
        return torch.stack(outputs, dim=0).permute(1, 0, 2, 3, 4)