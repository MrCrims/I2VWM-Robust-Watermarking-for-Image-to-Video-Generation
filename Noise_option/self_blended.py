import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from Noise_option.jpeg import Jpeg
import random
from Noise_option.cropout import get_random_rectangle_inside
import numpy as np


class SelfBlend(nn.Module):
    def __init__(self) -> None:
        super(SelfBlend,self).__init__()
        self.jpeg = Jpeg(Q=75,subsample=2)
        ratio = random.uniform(0.6,0.9)
        self.height_ratio_range = np.sqrt(ratio)
        self.width_ratio_range = np.sqrt(ratio)
    
    def forward(self,encoded_images,images):
        brightness = random.uniform(0.0,1.0)
        contrast = random.uniform(0.5,1.5)
        hue = random.uniform(-0.5,0.5)
        saturation = random.uniform(1,2)
        scale = random.uniform(0.5,.9)
        noised_images = TF.adjust_brightness(images,brightness_factor=brightness)
        noised_images = TF.adjust_contrast(noised_images,contrast_factor=contrast)
        noised_images = TF.adjust_hue(noised_images,hue_factor=hue)
        noised_images = TF.adjust_saturation(noised_images,saturation_factor=saturation)
        noised_images = F.interpolate(noised_images,scale_factor=(scale,scale))
        noised_images = F.interpolate(noised_images,size=(224,224))
        noised_images = self.jpeg(noised_images)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=encoded_images,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)

        cropout_mask = torch.zeros_like(encoded_images)
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
        blend_images = encoded_images * (1-cropout_mask) + noised_images * (cropout_mask)
        return blend_images