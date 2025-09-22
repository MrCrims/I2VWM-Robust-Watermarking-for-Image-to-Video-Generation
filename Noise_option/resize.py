import torch.nn as nn
import torch.nn.functional as F
import random



class Resize(nn.Module):
    """
    Resize the image.
    """
    def __init__(self, opt):
        super(Resize, self).__init__()
        resize_ratio_down = opt['noise']['Resize']['p']
        resize_ratio_down = random.uniform(resize_ratio_down[0],resize_ratio_down[1]) if isinstance(resize_ratio_down, list) else resize_ratio_down
        self.resize_ratio_down = resize_ratio_down
        self.interpolation_method = opt['noise']['Resize']['interpolation_method']

    def forward(self, wm_imgs, cover_img=None):
        
        #
        h,w = wm_imgs.shape[2:]
        noised_down = F.interpolate(
                                    wm_imgs,
                                    size=(int(h*self.resize_ratio_down), int(w*self.resize_ratio_down)),
                                    mode=self.interpolation_method
                                    )
        noised_up = F.interpolate(
                                    noised_down,
                                    size=(h,w),
                                    mode=self.interpolation_method
                                    )

        return noised_up


