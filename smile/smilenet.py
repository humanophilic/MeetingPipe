import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class resNetBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super(resNetBlock, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, x):
        out = F.relu(self.bn1a(self.conv1a(x)))
        out = self.conv2a(out)
        if self.stride == 1:
            residual = x
        else:
            residual = self.downsample(x)
        out = out + residual
        residual_ = out
        out = F.relu(self.outbna(out))

        out = F.relu(self.bn1b(self.conv1b(out)))
        out = self.conv2b(out)
        residual = residual_
        out = out + residual
        out = F.relu(self.outbnb(out))
        return out

class resNet18(nn.Module):
    def __init__(self):
        super(resNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3)),
            nn.BatchNorm2d(64, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        )
        self.block1 = resNetBlock(64, 64, stride=1)
        self.block2 = resNetBlock(64, 128, stride=2)
        self.block3 = resNetBlock(128, 256, stride=2)
        self.block4 = resNetBlock(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(3,3))

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.avgpool(out)
        return out

class selfattnLayer(nn.Module):
    def __init__(self, dropout):
        super(selfattnLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(128, 8, dropout=dropout)
        self.linear1 = nn.Linear(128, 512)
        self.linear2 = nn.Linear(512, 128)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x):
        x = x.transpose(0, 1)
        residual = x
        out = self.self_attn(x, x, x)[0]
        out = self.dropout1(out)
        out += residual
        out = self.norm1(out)
        residual = out
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout2(out)
        out = self.linear2(out)
        out = self.dropout3(out)
        out += residual
        out = self.norm2(out)
        out = out.transpose(0, 1)
        return out

class conv1D(nn.Module):
    def __init__(self):
        super(conv1D, self).__init__()
        self.conv1dnet = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1dnet(x)
        return out

class smileNet(nn.Module):
    def __init__(self, dropout):
        super(smileNet, self).__init__()
        self.resnet = resNet18()
        self.conv1d = conv1D()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.dropout = nn.Dropout(dropout)
        self.selfattn = selfattnLayer(dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        out = self.resnet(x)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = self.conv1d(out)
        out = out.transpose(1, 2)
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=out.shape[0])
        out = torch.cat((cls_tokens, out), dim=1)
        out = self.dropout(out)
        out = self.selfattn(out)
        out = out[:, 0]
        out = self.mlp(out)
        return out