import torch.nn as nn
import torch
import numpy as np

class Grayscale(nn.Module):
    """
    Converts the image to grayscale with a given probability.
    """
    def __init__(self, opt):
        """
        :param opt: Dictionary containing configuration options, including the probability of grayscaling.
        """
        super(Grayscale, self).__init__()
        self.p = opt['noise']['Grayscale']['p']  # Probability of converting to grayscale

    def forward(self, image):
        """
        :param image: Input image tensor of shape (B, C, H, W)
        :return: Grayscale image with probability p
        """
        if np.random.rand() < self.p:
            # Convert to grayscale using standard luminosity method
            grayscale_img = 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]
            return grayscale_img.expand_as(image)  # Expand to match input shape
        return image
