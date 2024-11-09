import torch
import torch.nn as nn
import torch.nn.functional as F

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

def random_motion(steps=16, initial_vector=None, alpha=0.2):
    if initial_vector is None:
        initial_vector = torch.randn(1, dtype=torch.cfloat)
    
    # Generate the random motion path
    motion = [torch.zeros_like(initial_vector)]
    for _ in range(steps):
        change = torch.randn(initial_vector.shape[0], dtype=torch.cfloat)
        initial_vector = initial_vector + change * alpha
        initial_vector /= initial_vector.abs().add(1e-8)
        motion.append(motion[-1] + initial_vector)
    
    motion = torch.stack(motion, -1)
    
    # Find bounding box
    real_min, _ = motion.real.min(dim=-1, keepdim=True)
    real_max, _ = motion.real.max(dim=-1, keepdim=True)
    imag_min, _ = motion.imag.min(dim=-1, keepdim=True)
    imag_max, _ = motion.imag.max(dim=-1, keepdim=True)

    # Scale motion to fit exactly in steps x steps
    real_scale = (steps - 1) / (real_max - real_min)
    imag_scale = (steps - 1) / (imag_max - imag_min)
    scale = torch.min(real_scale, imag_scale)
    
    real_shift = (steps - (real_max - real_min) * scale) / 2 - real_min * scale
    imag_shift = (steps - (imag_max - imag_min) * scale) / 2 - imag_min * scale
    
    scaled_motion = motion * scale + (real_shift + 1j * imag_shift)
    
    # Create kernel
    kernel = torch.zeros(initial_vector.shape[0], 1, steps, steps)
    
    # Fill kernel
    for s in range(steps + 1):
        v = scaled_motion[:, s]
        x = torch.clamp(v.real, 0, steps - 1)
        y = torch.clamp(v.imag, 0, steps - 1)
        
        ix = x.long()
        iy = y.long()
        
        vx = x - ix.float()
        vy = y - iy.float()
        
        for i in range(initial_vector.shape[0]):
            kernel[i, 0, iy[i], ix[i]] += (1-vx[i]) * (1-vy[i]) / steps
            if ix[i] + 1 < steps:
                kernel[i, 0, iy[i], ix[i]+1] += vx[i] * (1-vy[i]) / steps
            if iy[i] + 1 < steps:
                kernel[i, 0, iy[i]+1, ix[i]] += (1-vx[i]) * vy[i] / steps
            if ix[i] + 1 < steps and iy[i] + 1 < steps:
                kernel[i, 0, iy[i]+1, ix[i]+1] += vx[i] * vy[i] / steps

    # Normalize the kernel
    kernel /= kernel.sum(dim=(-1, -2), keepdim=True)
    
    return kernel

class RandomMotionBlur(nn.Module):
    """
    Apply random motion blur to input tensors.
    """
    def __init__(self, steps=17, alpha=0.2):
        """
        Initialize the RandomMotionBlur module.
        
        Args:
        - steps (int): Number of steps in the motion path
        - alpha (float): Controls the randomness of the motion path
        """
        super().__init__()
        self.steps = steps
        self.alpha = alpha
       
    def forward(self, x, return_kernel=False):
        """
        Apply random motion blur to the input tensor.
        
        Args:
        - x (torch.Tensor): Input tensor to be blurred
        - return_kernel (bool): If True, return both blurred tensor and kernel
        
        Returns:
        - y (torch.Tensor): Blurred tensor
        - m (torch.Tensor, optional): Blur kernel, if return_kernel is True
        """
        if x.dim() == 3:
            dim_3 = True
            x = x.unsqueeze(0)

        # Generate a random initial vector
        vector = torch.randn(x.shape[0], dtype=torch.cfloat) / 3
        vector.real /= 2
        
        # Create the motion blur kernel
        m = random_motion(self.steps, vector, alpha=self.alpha)
        
        # Pad the input tensor for convolution
        xpad = [m.shape[-1]//2+1] * 2 + [m.shape[-2]//2+1] * 2
        x = F.pad(x, xpad)
        
        # Pad the kernel to match input size
        mpad = [0, x.shape[-1]-m.shape[-1], 0, x.shape[-2]-m.shape[-2]]
        mp = F.pad(m, mpad)
        
        # Apply blur in the frequency domain
        fx = torch.fft.fft2(x)  # FFT of input
        fm = torch.fft.fft2(mp)  # FFT of kernel
        fy = fx * fm  # Multiplication in frequency domain
        y = torch.fft.ifft2(fy).real  # Inverse FFT to get blurred result
        
        # Crop the result to original size
        y = y[...,xpad[2]:-xpad[3], xpad[0]:-xpad[1]]

        if dim_3:
            y = y.squeeze(0)
        
        return y if not return_kernel else (y, m)