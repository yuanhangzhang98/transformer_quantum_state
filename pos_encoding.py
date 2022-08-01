import math
import numpy as np
import torch
import torch.nn as nn


class TQSPositionalEncoding1D(nn.Module):
    """
        A mixture of learnable and fixed positional encoding
        the first param_dim inputs have a learnable pe (parameter part)
        the rest have a sinusoidal pe (physical dimensions)
    """

    def __init__(self, d_model, param_dim, max_len=500, dropout=0):
        super(TQSPositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.param_embedding = nn.Parameter(
            torch.empty(param_dim, 1, d_model).normal_(std=0.02))  # (param_dim, 1, d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.get_default_dtype()).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.get_default_dtype()) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        self.pe = torch.cat([self.param_embedding, pe], dim=0)  # (param_dim+n, 1, d_model)

    def forward(self, x, system_size=None):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
            system_size: the size of the system. Not used here, only included to be consistent with the other positional
             encoders
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TQSPositionalEncoding2D(nn.Module):
    """
        A mixture of learnable and fixed positional encoding
        the first param_dim inputs have a learnable pe (parameter part)
        the rest have a sinusoidal pe (physical dimensions)
    """

    def __init__(self, d_model, param_dim, max_system_size=None, dropout=0):
        super(TQSPositionalEncoding2D, self).__init__()
        if max_system_size is None:
            max_system_size = [50, 50]
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        max_system_size = np.array(max_system_size).reshape(-1)
        self.max_system_size = max_system_size
        self.param_embedding = nn.Parameter(
            torch.empty(param_dim, 1, d_model).normal_(std=0.02))  # (param_dim, 1, d_model)

        assert len(max_system_size) == 2

        x, y = max_system_size
        channels = int(np.ceil(d_model / 4) * 2)
        div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.get_default_dtype()) * (
                -math.log(10000.0) / channels))  # channels/2
        pos_x = torch.arange(x, dtype=div_term.dtype).unsqueeze(1)  # (nx, 1)
        pos_y = torch.arange(y, dtype=div_term.dtype).unsqueeze(1)  # (ny, 1)
        sin_inp_x = pos_x * div_term  # (nx, channels/2)
        sin_inp_y = pos_y * div_term  # (ny, channels/2)
        emb_x = torch.zeros(x, channels)
        emb_y = torch.zeros(y, channels)
        emb_x[:, 0::2] = sin_inp_x.sin()
        emb_x[:, 1::2] = sin_inp_x.cos()
        emb_y[:, 0::2] = sin_inp_y.sin()
        emb_y[:, 1::2] = sin_inp_y.cos()
        pe = torch.zeros((x, y, 2 * channels))
        pe[:, :, :channels] = emb_x.unsqueeze(1)
        pe[:, :, channels:] = emb_y  # (x, y, 2*channels)
        pe = pe[:, :, :d_model]  # (x, y, d_model)
        pe = pe.unsqueeze(2)  # (x, y, 1, d_model)

        self.pe = pe  # (x, y, 1, d_model)

    def forward(self, x, system_size=None):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
            system_size: the size of the system. Default: None, uses max_system_size
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        if system_size is None:
            system_size = self.max_system_size
        pe = self.pe[:system_size[0], :system_size[1]].reshape(-1, 1, self.d_model)
        pe = torch.cat([self.param_embedding, pe], dim=0)  # (param_dim+n, 1, d_model)
        x = x + pe[:x.size(0)]
        return self.dropout(x)
