import torch.nn as nn
import torch
import numpy as np


class HorizontalFlip(nn.Module):
    """
    Randomly flips the image horizontally with a given probability.
    """
    def __init__(self, opt):
        """
        :param opt: Dictionary containing configuration options, including the probability of flipping.
        """
        super(HorizontalFlip, self).__init__()
        self.p = opt['noise']['Flip']['p']  # Probability of flipping

    def forward(self, image):
        """
        :param image: Input image tensor of shape (B, C, H, W)
        :return: Horizontally flipped image with probability p
        """
        if np.random.rand() < self.p:
            return torch.flip(image, dims=[3])  # Flip along width dimension
        return image


