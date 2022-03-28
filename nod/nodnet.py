import torch.nn as nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8): 
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class RNN(nn.Module):
    def __init__(self, gpu_id, conv_hidden_size, rnn_hidden_size, num_layers, dropout):
        super(RNN, self).__init__()
        self.gpu_id = gpu_id
        self.num_layers = num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.gru = nn.GRU(conv_hidden_size, rnn_hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(rnn_hidden_size, 2)
        self.conv = nn.Conv1d(1, conv_hidden_size, 3, stride=1)
        self.bn = nn.BatchNorm1d(conv_hidden_size)
        self.avgpool = nn.AvgPool1d(3, stride=1)
        self.se = SELayer(rnn_hidden_size, reduction=8)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x = x.transpose(1,2)
        # out = self.conv(x)
        # out = self.bn(out)
        # out = self.silu(out)
        # out = self.avgpool(out)
        # out = out.transpose(1,2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.rnn_hidden_size).cuda(self.gpu_id)
        out, _ = self.gru(x, h0)
        residual = out
        out = out.transpose(1,2)
        out = self.se(out)
        out = out.transpose(1,2)
        out += residual
        out = self.relu(out)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out