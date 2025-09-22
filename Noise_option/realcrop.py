import torch
import torch.nn as nn
from Noise_option.cropout import get_random_rectangle_inside
import numpy as np
import random




class RealCrop(nn.Module):
    """
    Keep the value of only one random rectangular area and set the value of other areas to 0
    """
    def __init__(self, opt):
        super(RealCrop, self).__init__()
        #
        ratio = opt['noise']['RealCrop']['p']  # 9% of total pixel
        ratio = random.uniform(ratio[0],ratio[1]) if isinstance(ratio, list) else ratio
        #
        self.height_ratio_range = np.sqrt(ratio)
        self.width_ratio_range = np.sqrt(ratio)

    def forward(self, encoded_img, cover_img=None):
        h,w = encoded_img.shape[2:]

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=encoded_img,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        noised_img = encoded_img[:, :, h_start:h_end, w_start:w_end]
        noised_img = nn.functional.interpolate(noised_img,size=(h,w))


        return  noised_img

