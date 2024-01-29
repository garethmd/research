import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms as transforms


class TransformPipeline:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, batch: torch.Tensor):
        for transform in self.transforms:
            batch = transform(batch)
        return batch
    
class BatchToTensor:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch: np.array):
        """
        Args:
            batch (np.array): expects a numpy array
        Returns:
            torch.Tensor: torch tensor on device
        """
        return torch.from_numpy(batch).to(self.device)
    
class BatchCrop:
    def __init__(self, size=(84,84)):
        self.size = size
    
    def __call__(self, batch: torch.Tensor):
        """ crop 17 pixels from top and bottom
        Args:
            batch (torch.Tensor): expects batch to be of shape (B, C, H, W)
        Returns:
            torch.Tensor: batch of shape (B, C, H, W)
        """
        return batch[:, :,  17:17+self.size[0], :] # crop 17 pixels from top and bottom
    
class BatchDownsample:
    def __init__(self, size=(110,84)):
        self.size = size
    
    def __call__(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): expects batch to be of shape (B, C, H, W)
        Returns:
            torch.Tensor: batch of shape (B, C, H, W)
        """
        orig_dtype = batch.dtype
        result = nn.functional.interpolate(batch.float(), size=self.size,  mode='bilinear')
        return result.type(orig_dtype)
    

class BatchGrayscale:
    def __init__(self):
        self.transforms = transforms.Grayscale()

    def __call__(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): expects batch to be of shape (B, H, W, C)
        Returns:
            torch.Tensor: batch of shape (B, C, H, W)
        """
        grey_pt_stack = torch.stack([self.transforms(s.permute(2,0,1)) for s in batch])
        return grey_pt_stack