import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from io import BytesIO

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
    
class RandomMotionBlur(nn.Module):
    """
    Apply random motion blur to input tensors using convolution.
    """
    def __init__(self, ks=17, alpha=0.2):
        """
        Initialize the RandomMotionBlur module.
        
        Args:
        - ks (int): Kernel size = number of steps in the motion path
        - alpha (float): Controls the randomness of the motion path
        """
        super().__init__()
        self.ks = ks
        self.alpha = alpha
    
    def _generate_motion_kernel(self, initial_vector=None, alpha=0.2):
        if initial_vector is None:
            initial_vector = torch.randn(1, dtype=torch.cfloat)
        
        # Generate the random motion path
        motion = [torch.zeros_like(initial_vector)]
        for _ in range(self.ks):
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
        real_scale = (self.ks - 1) / (real_max - real_min)
        imag_scale = (self.ks - 1) / (imag_max - imag_min)
        scale = torch.min(real_scale, imag_scale)
        
        real_shift = (self.ks - (real_max - real_min) * scale) / 2 - real_min * scale
        imag_shift = (self.ks - (imag_max - imag_min) * scale) / 2 - imag_min * scale
        
        scaled_motion = motion * scale + (real_shift + 1j * imag_shift)
        
        # Create kernel
        kernel = torch.zeros(initial_vector.shape[0], 1, self.ks, self.ks)
        
        # Fill kernel
        for s in range(self.ks + 1):
            v = scaled_motion[:, s]
            x = torch.clamp(v.real, 0, self.ks - 1)
            y = torch.clamp(v.imag, 0, self.ks - 1)
            
            ix = x.long()
            iy = y.long()
            
            vx = x - ix.float()
            vy = y - iy.float()
            
            for i in range(initial_vector.shape[0]):
                kernel[i, 0, iy[i], ix[i]] += (1-vx[i]) * (1-vy[i]) / self.ks
                if ix[i] + 1 < self.ks:
                    kernel[i, 0, iy[i], ix[i]+1] += vx[i] * (1-vy[i]) / self.ks
                if iy[i] + 1 < self.ks:
                    kernel[i, 0, iy[i]+1, ix[i]] += (1-vx[i]) * vy[i] / self.ks
                if ix[i] + 1 < self.ks and iy[i] + 1 < self.ks:
                    kernel[i, 0, iy[i]+1, ix[i]+1] += vx[i] * vy[i] / self.ks

        # Normalize the kernel
        kernel /= kernel.sum(dim=(-1, -2), keepdim=True)
        
        return kernel
    
    def forward(self, x, return_kernel=False):
        """
        Apply random motion blur to the input tensor using convolution.
        
        Args:
        - x (torch.Tensor): Input tensor to be blurred [B,C,H,W] or [C,H,W]
        - return_kernel (bool): If True, return both blurred tensor and kernel
        
        Returns:
        - y (torch.Tensor): Blurred tensor
        - kernel (torch.Tensor, optional): Blur kernel, if return_kernel is True
        """
        dim_3 = False
        if x.dim() == 3:
            dim_3 = True
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Generate initial vector and kernel
        vector = torch.randn(batch_size, dtype=torch.cfloat, device=x.device) / 3
        vector.real /= 2
        kernel = self._generate_motion_kernel(initial_vector=vector, alpha=self.alpha)

        # Replicate kernel for each input channel
        kernel = kernel.repeat(3, 1, 1, 1) #repeat(1, x.shape[1], 1, 1)

        # Apply convolution
        padding = (self.ks // 2, self.ks // 2)
        y = F.conv2d(
            input=x,
            weight=kernel,
            padding=padding,
            groups=3
        )
        
        if dim_3:
            y = y.squeeze(0)
            kernel = kernel.squeeze(0)
        
        return y if not return_kernel else (y, kernel)
    
class CircularPSFBlur(nn.Module):
    """
    Apply out of focus blur using circular PSF convolution.
    """
    def __init__(self, kernel_size=13, radius=None):
        """
        Initialize the CircularPSFBlur module.
        
        Args:
        - kernel_size (int): Size of the PSF kernel
        - radius (float, optional): Radius of the circular PSF. If None, uses 40% of kernel_size
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.radius = radius if radius is not None else 0.4 * kernel_size
    
    def _generate_circular_psf(self, batch_size=1):
        """
        Generate circular PSF kernel.
        
        Args:
        - batch_size (int): Number of kernels to generate
        
        Returns:
        - kernel (torch.Tensor): Circular PSF kernel
        """
        # Create coordinate grids
        x = torch.linspace(-self.kernel_size/2, self.kernel_size/2, self.kernel_size)
        y = torch.linspace(-self.kernel_size/2, self.kernel_size/2, self.kernel_size)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
        
        # Create circular mask
        radius_map = torch.sqrt(x_grid**2 + y_grid**2)
        kernel = (radius_map <= self.radius).float()
        
        # Optional: Smooth the edges slightly for more realistic blur
        sigma = 0.5
        if sigma > 0:
            kernel = torch.exp(-((radius_map - self.radius)**2) / (2 * sigma**2))
            kernel = kernel * (radius_map <= (self.radius + 2*sigma)).float()
        
        # Normalize the kernel
        kernel = kernel / kernel.sum()
        
        # Add batch and channel dimensions
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(batch_size, 1, 1, 1)
        
        return kernel
    
    def forward(self, x, return_kernel=False):
        """
        Apply out of focus blur to the input tensor using convolution.
        
        Args:
        - x (torch.Tensor): Input tensor to be blurred [B,C,H,W] or [C,H,W]
        - return_kernel (bool): If True, return both blurred tensor and kernel
        
        Returns:
        - y (torch.Tensor): Blurred tensor
        - kernel (torch.Tensor, optional): Blur kernel, if return_kernel is True
        """
        dim_3 = False
        if x.dim() == 3:
            dim_3 = True
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Generate kernel
        kernel = self._generate_circular_psf(batch_size).to(x.device)
        
        # Replicate kernel for each input channel (RGB)
        kernel = kernel.repeat(3, 1, 1, 1)
        
        # Apply convolution
        padding = (self.kernel_size // 2, self.kernel_size // 2)
        y = F.conv2d(
            input=x,
            weight=kernel,
            padding=padding,
            groups=3  # Process each channel independently
        )
        
        if dim_3:
            y = y.squeeze(0)
            kernel = kernel.squeeze(0)
        
        return y if not return_kernel else (y, kernel)
    
class JPEGArtifacts(nn.Module):
    """Add JPEG compression artifacts to tensor images.
    
    Args:
        quality (int): JPEG quality factor (0-100, default: 50)
        p (float): Probability of applying the compression (default: 1.0)
    """
    def __init__(self, quality=50, p=1.0):
        super().__init__()
        self.quality = quality
        self.p = p
        
    def forward(self, tensor):
        if torch.rand(1) < self.p:
            # Convert to PIL Image
            img = TF.to_pil_image(tensor)
            buffer = BytesIO()
            
            # Apply JPEG compression
            img.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            compressed = Image.open(buffer)
            
            # Convert back to tensor
            return TF.to_tensor(compressed)
        return tensor
        
    def __repr__(self):
        return f"{self.__class__.__name__}(quality={self.quality}, p={self.p})"