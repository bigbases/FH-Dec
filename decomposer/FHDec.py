import math
import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Frequency-based Decomposer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.kernel_size = configs.kernel_size
        self.freq_len = configs.freq_len
        
        self.series_decomp = series_decomp(configs.kernel_size)
    
    def forward(self, x):
        # Use only target series
        x = x[:, :, -1:]
        
        # Seasonal-Trend decomposition
        seasonal, trend = self.series_decomp(x)
        
        # Frequency-based decomposition
        seasonal_windows = seasonal.unfold(1, self.freq_len, 1).squeeze(2)
        spectrums = torch.fft.fft(seasonal_windows, dim=1)[:, :, :self.freq_len//2 + 1]
        real_part = spectrums.real
        imag_part = spectrums.imag
        
        # Truncation
        x = x[:, self.freq_len-1:, :]
        seasonal = seasonal[:, self.freq_len-1:, :]
        trend = trend[:, self.freq_len-1:, :]
        
        # Concatenation
        x = torch.cat([real_part, imag_part, seasonal, trend, x], dim=2)

        return x