import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

class Combined_model_t(nn.Module):
    def __init__(self, TRAN, FC):
        super(Combined_model_t, self).__init__()
        self.TRAN = TRAN
        self.FC = FC
        # self.classifier = nn.Linear(4, 2)

    def forward(self, src, tgt_mask, key_mask):

        src = self.TRAN.encoder(src.transpose(1, 2)) # (1, 24, 256)
        src = self.TRAN.pos_encoder(src) # (1, 24, 256)
        x = self.TRAN.transformer_encoder(src, tgt_mask, key_mask).transpose(1, 2) # after transpose: (1, 256, 24)
        x = self.FC(x) # (bs, T, 2)
        x = torch.mean(x, dim=1) # (bs, )
        x = nn.Softmax(dim=-1)(x)
        return x
    
    def get_tgt_mask(self, size):
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = ~torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        # mask = mask.float()
        # mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        # mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # tensor([[False,  True,  True,  True,  True],
        # [False, False,  True,  True,  True],
        # [False, False, False,  True,  True],
        # [False, False, False, False,  True],
        # [False, False, False, False, False]])

        return mask

class Combined_model_l(nn.Module):
    def __init__(self, RNN, FC):
        super(Combined_model_l, self).__init__()
        self.rnn = RNN.rnn
        self.FC = FC
        self.idrop = RNN.idrop
        self.lockdrop = RNN.lockdrop
        # self.classifier = nn.Linear(4, 2)

    def forward(self, td_input, td_lengths):
        # td should be (4, 40, 200)
        # td_inpus is (4, 200, 40), length is [32, 34, 38, 40]
        # h0 = torch.zeros(self.layer_dim, td_input.size(0), self.hidden_dim).requires_grad_()
        if self.idrop >0: 
            td_input = self.lockdrop(td_input, self.idrop)
        packed_td = pack_padded_sequence(td_input, td_lengths, batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.rnn(packed_td)
        # padded out shoudl be (4, 40, hidden_dim)
        padded_out, _ = pad_packed_sequence(out, batch_first=True)
        # padded_out = padded_out[:, -1, :]
        # final output is (4, 40, output_classes)
        padded_out = self.FC(padded_out)
        x = torch.mean(padded_out, dim=1) # (bs, )
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
        
class FCNet_l(nn.Module):
    def __init__(self, num_inputs, num_channels, dropout, reluslope, output_class):
        super(FCNet_l, self).__init__()
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
        # x = x.contiguous().transpose(1, 2)
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

# Transformer models
class PositionalEncoding(nn.Module):

    "Implement the PE function."
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Trans_encoder(nn.Module):
    def __init__(self, feature_dim, d_model, nhead, d_hid, nlayers, out_dim, dropout):
        super(Trans_encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(feature_dim, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, out_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask, key_mask):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, feature_dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, 1]
        """
        src = src.contiguous().transpose(1, 2) #(6, 24, 182)
        src = self.encoder(src) # (6, 24, 256)
        src = self.pos_encoder(src) # (6, 24, 256)
        output = self.transformer_encoder(src, src_mask, key_mask) # (6, 24, 256)
        output = self.decoder(output) # (6, 24, 1)
        return output
    
    def get_tgt_mask(self, size):
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = ~torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        # mask = mask.float()
        # mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        # mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # tensor([[False,  True,  True,  True,  True],
        # [False, False,  True,  True,  True],
        # [False, False, False,  True,  True],
        # [False, False, False, False,  True],
        # [False, False, False, False, False]])

        return mask

# RNN 
class LockedDropout(nn.Module):
    '''
    Dropout that is consistent on the sequence dimension.
    '''
    def forward(self, x, dropout=0.5):
        if not self.training:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask/torch.max(mask)
        mask = mask.expand_as(x)
        return mask * x

class RecurrentModel(nn.Module):
    '''
    Recurrent model for time series data.
    '''
    def __init__(self, cell='RNN', input_dim=200, hidden_dim=10, layer_dim=3, output_dim=2, dropout_prob=0.2, idrop=0, activation='tanh'):
        super(RecurrentModel, self).__init__()

        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout_prob = dropout_prob
        self.output_dim = output_dim
        self.cell = cell
        self.nonlinearity = activation
        self.idrop = idrop 

        # Defining the number of layers and the nodes in each layer
        if isinstance(cell, str):
            if self.cell.upper() == 'RNN':
                self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout_prob, nonlinearity=self.nonlinearity)
            elif self.cell.upper() == 'LSTM':
                self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout_prob)
            elif self.cell.upper() == 'GRU':
                self.rnn = nn.GRU(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=self.dropout_prob)
            else:
                raise Exception('Only GRU, LSTM and RNN are supported as cells.')

        # # encode time-incariant layer 
        self.lockdrop = LockedDropout()
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, td_input, td_lengths):
        # td should be (4, 40, 200)
        # td_inpus is (4, 200, 40), length is [32, 34, 38, 40]
        # h0 = torch.zeros(self.layer_dim, td_input.size(0), self.hidden_dim).requires_grad_()
        if self.idrop >0: 
            td_input = self.lockdrop(td_input, self.idrop)
        packed_td = pack_padded_sequence(td_input, td_lengths, batch_first=True, enforce_sorted=False)

        if self.cell.upper() == 'LSTM':
            # c0 = torch.zeros(self.layer_dim, td_input.size(0), self.hidden_dim).requires_grad_()
            out, (hn, cn) = self.rnn(packed_td)
        else:
            out, h0 = self.rnn(packed_td)
        # padded out shoudl be (4, 40, hidden_dim)
        padded_out, _ = pad_packed_sequence(out, batch_first=True)
        # padded_out = padded_out[:, -1, :]
        # final output is (4, 40, output_classes)
        padded_out = self.fc(padded_out)
        return padded_out