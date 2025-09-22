import torch.nn as nn
import torchvision.transforms as transforms
import kornia as K
import random

#
class ColorJitter(nn.Module):
    """
    
    """
    def __init__(self, opt, distortion):
        super(ColorJitter, self).__init__()
        #
        brightness   = opt['noise']['Brightness']['f']
        brightness = random.uniform(brightness[0],brightness[1]) if isinstance(brightness, list) else brightness
        contrast     = opt['noise']['Contrast']['f']
        contrast = random.uniform(contrast[0],contrast[1]) if isinstance(contrast, list) else contrast
        saturation   = opt['noise']['Saturation']['f']
        saturation = random.uniform(saturation[0],saturation[1]) if isinstance(saturation, list) else saturation
        hue          = opt['noise']['Hue']['f']
        hue = random.uniform(hue[0],hue[1]) if isinstance(hue, list) else hue
        #
        if distortion == 'Brightness':
            self.transform = transforms.ColorJitter(brightness=brightness)
        if distortion == 'Contrast':
            self.transform = transforms.ColorJitter(contrast=contrast)
        if distortion == 'Saturation':
            self.transform = transforms.ColorJitter(saturation=saturation)
        if distortion == 'Hue':
            self.transform = transforms.ColorJitter(hue=hue)

    def forward(self, watermarked_img, cover_img=None):
        #
        watermarked_img = (watermarked_img + 1 ) / 2   # [-1, 1] -> [0, 1]
        #
        ColorJitter = self.transform(watermarked_img)
        #
        ColorJitter = (ColorJitter * 2) - 1  # [0, 1] -> [-1, 1]

        return ColorJitter


