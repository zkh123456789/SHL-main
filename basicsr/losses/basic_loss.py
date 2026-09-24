import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """Enhanced L1 loss with gradient and frequency components for remote sensing.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        gradient_weight (float): Weight for gradient loss component. Default: 0.0.
        frequency_weight (float): Weight for frequency loss component. Default: 0.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        kernel_size (int): Kernel size for Gaussian blur in frequency loss. Default: 5.
    """

    def __init__(self, loss_weight=1.0, gradient_weight=0.0, frequency_weight=0.0, 
                 reduction='mean', kernel_size=5):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.gradient_weight = gradient_weight
        self.frequency_weight = frequency_weight
        self.reduction = reduction
        self.kernel_size = kernel_size
        
        # Register Sobel filters as buffers for gradient calculation
        if self.gradient_weight > 0:
            # Create Sobel filters for 3-channel input
            sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], 
                                  requires_grad=False).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], 
                                  requires_grad=False).view(1, 1, 3, 3)
            
            # Expand Sobel filters to handle 3-channel input
            self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))
            self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))

    def calculate_gradient_loss(self, pred, target, weight=None):
        """Calculate gradient loss component."""
        # Calculate gradients for predicted image
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        grad_pred = torch.abs(grad_pred_x) + torch.abs(grad_pred_y)
        
        # Calculate gradients for target image
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        grad_target = torch.abs(grad_target_x) + torch.abs(grad_target_y)
        
        # Calculate gradient loss
        loss = F.l1_loss(grad_pred, grad_target, reduction='none')
        
        if weight is not None:
            loss = loss * weight
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def calculate_frequency_loss(self, pred, target, weight=None):
        """Calculate frequency loss component."""
        # Separate low-frequency components using Gaussian blur
        low_freq_pred = TF.gaussian_blur(pred, kernel_size=self.kernel_size)
        low_freq_target = TF.gaussian_blur(target, kernel_size=self.kernel_size)
        
        # High-frequency components = original - low-frequency
        high_freq_pred = pred - low_freq_pred
        high_freq_target = target - low_freq_target
        
        # Calculate high-frequency loss
        loss = F.l1_loss(high_freq_pred, high_freq_target, reduction='none')
        
        if weight is not None:
            loss = loss * weight
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        # Calculate base L1 loss
        base_loss = l1_loss(pred, target, weight, reduction=self.reduction)
        total_loss = self.loss_weight * base_loss
        
        # Add gradient loss if weight > 0
        if self.gradient_weight > 0:
            gradient_loss = self.calculate_gradient_loss(pred, target, weight)
            total_loss += self.gradient_weight * gradient_loss
        
        # Add frequency loss if weight > 0
        if self.frequency_weight > 0:
            frequency_loss = self.calculate_frequency_loss(pred, target, weight)
            total_loss += self.frequency_weight * frequency_loss
            
        return total_loss
