import torch

# Q: [B, L, Q, H, E]->[8, 384, 1, 8, 64]
class TriangularCausalMask:
    def __init__(self, mask_shape, device):
        with torch.no_grad():
            _mask = torch.ones(mask_shape, dtype=torch.bool)
            self._mask = torch.triu(_mask, diagonal=1).to(device)  # 在最后两个维度上填充

    @property
    def mask(self):
        return self._mask
