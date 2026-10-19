import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dropout=0.1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return out + shortcut

class CifarNeural(nn.Module):
    def __init__(self, blocks_per_stage=(4, 4, 4), dropout=0.1):
        super().__init__()

        self.stem = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.stage1 = self._make_stage(64, 64, blocks_per_stage[0], stride=1, dropout=dropout)
        self.stage2 = self._make_stage(64, 128, blocks_per_stage[1], stride=2, dropout=dropout)
        self.stage3 = self._make_stage(128, 256, blocks_per_stage[2], stride=2, dropout=dropout)

        self.final_bn = nn.BatchNorm2d(256)
        self.final_relu = nn.ReLU(inplace=True)

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(256, 10)

    def _make_stage(self, in_c, out_c, num_blocks, stride, dropout):
        layers = []
        layers.append(ResBlock(in_c, out_c, stride=stride, dropout=dropout))
        
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_c, out_c, stride=1, dropout=dropout))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.final_bn(x)
        x = self.final_relu(x)

        x = self.globalpool(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x