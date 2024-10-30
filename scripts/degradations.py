import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    """Add Gaussian noise to tensor images.
    
    Args:
        mean (float): Mean of the Gaussian noise (default: 0)
        std (float): Standard deviation of the Gaussian noise (default: 0.1)
        p (float): Probability of applying the noise (default: 0.5)
    """
    def __init__(self, mean=0, std=0.1, p=1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p
        
    def forward(self, tensor):
        if torch.rand(1) < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})"