import torch
import math
import numpy as np
import torch.nn as nn
import torch.fft as fft  # 1.5<=torch.version<=1.7
from utils.tools import complex_standardization


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #  x: [4, 672, 8, 12]
        return self.pe[:, :x.size(1)]


def positional_embedding(x, d_model, max_len=5000):
    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe[:, :x]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=(1, 3), padding=(0, 1), padding_mode='reflect')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        #  [B, L, P, E]
        #  [4, 672, 8, 50]->[4, 50, 8, 672]->[4, 672, 8, 50]
        x = x.permute(0, 3, 2, 1)
        x = self.tokenConv(x)
        x = x.permute(0, 3, 2, 1)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


def convert_freqcube_2_amp_phase(freqcube):
    return freqcube.abs(), freqcube.angle()


def convert_amp_phase_2_freqcube(amp, phase):
    return torch.complex(real=amp * torch.cos(phase), imag=amp * torch.sin(phase))


def convert_freqcube_2_real_img(freqcube):
    return freqcube.real, freqcube.imag


def convert_real_img_2_freqcube(real, imag):
    return torch.complex(real=real, imag=imag)


class FreqEmbedding(nn.Module):
    def __init__(self, args):
        super(FreqEmbedding, self).__init__()
        self.window_length = None
        self.args = args

    def forward(self, seq):
        seq_freq_cube = self.convert_seq_2_feqcube(seq, self.args)
        seq_freq_cube, _, _ = complex_standardization(seq_freq_cube, dims=(1, 2), method='Normalization')
        seq_freq_amp, _ = convert_freqcube_2_amp_phase(seq_freq_cube)
        return seq_freq_amp

    def convert_seq_2_feqcube(self, sequence, args):
        step = sequence.size(1)  # sequence_length
        window = self.choose_window(args).unsqueeze(-1).repeat(1, sequence.size(-1)).to(self.args.device)
        m = nn.ReflectionPad1d(int(args.window_length / 2))
        sequence = m(sequence.transpose(-1, -2)).transpose(-1, -2).to(self.args.device)
        cat_seq = (sequence[:, 0:args.window_length, :] * window).unsqueeze(1).to(self.args.device)
        for i in range(1, step):
            cat_seq = torch.cat((cat_seq, (sequence[:, i:i + args.window_length, :] * window).unsqueeze(1)), dim=1)
        return fft.rfft(cat_seq, dim=-2)

    # @return : 1-D tensor
    def choose_window(self, args):
        assert args.window in ['hamming', 'hanning', 'rect']
        if args.window == 'hamming':
            window = np.array(
                [0.54 - 0.46 * np.cos(2 * np.pi * n / (args.window_length - 1)) for n in range(args.window_length)])
        elif args.window == 'hanning':
            window = np.array(
                [0.5 - 0.5 * np.cos(2 * np.pi * n / (args.window_length - 1)) for n in range(args.window_length)])
        else:
            window = np.ones(args.window_length)
        return torch.tensor(window).to(torch.float32)


class EncEmbedding(nn.Module):
    def __init__(self, args):
        super(EncEmbedding, self).__init__()
        self.c_in = int(args.window_length / 2) + 2
        self.d_model = args.d_model
        self.embed_type = args.embed_type
        self.freq = args.freq
        self.dropout = args.dropout

        self.freq_embedding = FreqEmbedding(args)
        self.value_embedding = TokenEmbedding(c_in=self.c_in, d_model=self.d_model)
        self.position_embedding = PositionalEmbedding(d_model=self.d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=self.d_model, embed_type=self.embed_type,
                                                    freq=self.freq) if self.embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=self.d_model, embed_type=self.embed_type, freq=self.freq)

        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x, x_mark):
        freq_x = self.freq_embedding(x)  # [4, 672, 49, 8]
        x = torch.cat((x.unsqueeze(-2), freq_x), dim=-2).transpose(-1, -2)  # [4, 672, 8, 50]
        val_emb = self.value_embedding(x)
        pos_emb = self.position_embedding(x).unsqueeze(-2)
        temp_emb = self.temporal_embedding(x_mark).unsqueeze(-2)
        x = pos_emb + temp_emb + val_emb
        return self.dropout(x)


class DecEmbedding(nn.Module):
    def __init__(self, args):
        super(DecEmbedding, self).__init__()
        self.c_in = 1
        self.d_model = args.d_model
        self.embed_type = args.embed_type
        self.freq = args.freq
        self.dropout = args.dropout

        self.value_embedding = TokenEmbedding(c_in=self.c_in, d_model=self.d_model)
        self.position_embedding = PositionalEmbedding(d_model=self.d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=self.d_model, embed_type=self.embed_type,
                                                    freq=self.freq) if self.embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=self.d_model, embed_type=self.embed_type, freq=self.freq)

        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x, x_mark):
        x = x.unsqueeze(-1)
        val_emb = self.value_embedding(x)
        pos_emb = self.position_embedding(x).unsqueeze(-2)
        temp_emb = self.temporal_embedding(x_mark).unsqueeze(-2)
        x = val_emb + pos_emb + temp_emb
        return self.dropout(x)
