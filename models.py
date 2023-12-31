import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Combined_model(nn.Module):
    def __init__(self, TCN, FC):
        super(Combined_model, self).__init__()
        self.TCN = TCN
        self.FC = FC
        # self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        x = self.TCN(x)
        x = self.FC(x) # (bs, T, 2)
        x = torch.mean(x, dim=1) # (bs, )
        x = nn.Softmax(dim=-1)(x)
        return x


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_channels, dropout, reluslope, output_class):
        super(FCNet, self).__init__()
        self.num_inputs = num_inputs # 256 for Transformer 
        # num_channels [128]
        layers = []
        for i in range(len(num_channels) ):
            composite_in = self.num_inputs if i == 0 else num_channels[i-1]
            layers += [nn.Linear(composite_in, num_channels[i])]
            layers += [nn.LeakyReLU(reluslope)]
            layers += [nn.Dropout(dropout)]

        layers += [nn.Linear(num_channels[-1], output_class)]

        self.FC = nn.Sequential(*layers)

    def forward(self, x):
        # x is (6, 256, 24) or (6, 256, 40) 
        x = x.contiguous().transpose(1, 2)
        # (6, 24, 256)
        x = self.FC(x)
        # (6, 24, num_classes)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

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


class TemporalConv(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, output_class=2):
        super(TemporalConv, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Sequential(
        nn.Linear(num_channels[-1], 128),
        # nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        # nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, output_class)
        )


    def forward(self, x):
        x = self.network(x)
        x = x.contiguous().transpose(1, 2)
        x = self.linear(x)
        return x