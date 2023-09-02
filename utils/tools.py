import torch
import numpy as np


class StandardScaler:
    def __init__(self):
        self.std = None
        self.mean = None
        self.max = 1.
        self.min = 0.

    # def fit(self, data):
    #     self.max = np.apply_along_axis(np.max, 0, data)
    #     self.min = np.apply_along_axis(np.min, 0, data)
    #
    # def transform(self, data):
    #     stand = (data-self.min)/(self.max-self.min)
    #     return stand
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std


# MSE计算
def MSE(target, predict):
    return ((target - predict) ** 2).mean()


# RMSE计算
def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# MAE计算
def MAE(target, predict):
    return (abs(target - predict)).mean()


# 在指定的两个维度上对复数张量的幅值标准化。
# value' = (value - A) / B; ->> value = B * value' + A

def complex_standardization(complex_tensor, dims, method='Normalization'):
    amp = complex_tensor.abs()
    phase = complex_tensor.angle()

    if method == 'Normalization':
        A = amp.min(dim=dims[0], keepdims=True).values.min(dim=dims[1], keepdims=True).values
        B = amp.max(dim=dims[0], keepdims=True).values.max(dim=dims[1], keepdims=True).values - A
        amp = (amp - A) / B
    else:
        A = amp_avg = amp.mean(dim=dims, keepdims=True)
        B = amp_std = amp.std(dim=dims, keepdims=True)
        amp = (amp - amp_avg) / amp_std
    return torch.complex(real=amp * torch.cos(phase), imag=amp * torch.sin(phase)), A, B
