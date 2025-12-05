import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchvision import models

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Dilated Causal Conv 1
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Dilated Causal Conv 2
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CSIEncoder(nn.Module):
    def __init__(self, input_channels=1, output_dim=128):
        super(CSIEncoder, self).__init__()
        # TCN Architecture based on diagram (Dilated Causal Convs)
        # Input: [Batch, 1, Length]
        # We use a 4-level TCN to extract temporal features
        num_channels = [32, 64, 128, 128]
        self.tcn = TemporalConvNet(num_inputs=input_channels, num_channels=num_channels, kernel_size=3, dropout=0.2)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: [Batch, Length] -> [Batch, 1, Length]
        x = x.unsqueeze(1)
        y = self.tcn(x)
        # Global Average Pooling over time dimension
        y = torch.mean(y, dim=2)
        y = self.fc(y)
        return y

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super(ImageEncoder, self).__init__()
        # ResNet50 Backbone (Matches the x3, x4, x6, x3 blocks in diagram)
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # Output: [Batch, 2048, 1, 1]
        self.fc = nn.Linear(2048, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FusionModel(nn.Module):
    def __init__(self, num_classes=8, mode='fusion'):
        super(FusionModel, self).__init__()
        self.mode = mode
        self.csi_encoder = CSIEncoder(input_channels=1, output_dim=128)
        self.img_encoder = ImageEncoder(embed_dim=128)
        
        if mode == 'fusion':
            # Concatenate features: 128 + 128 = 256
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        elif mode == 'img_only':
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        elif mode == 'csi_only':
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

    def forward(self, csi, img):
        if self.mode == 'fusion':
            csi_feat = self.csi_encoder(csi)
            img_feat = self.img_encoder(img)
            combined = torch.cat((csi_feat, img_feat), dim=1)
            out = self.classifier(combined)
            return out
        elif self.mode == 'img_only':
            img_feat = self.img_encoder(img)
            out = self.classifier(img_feat)
            return out
        elif self.mode == 'csi_only':
            csi_feat = self.csi_encoder(csi)
            out = self.classifier(csi_feat)
            return out
