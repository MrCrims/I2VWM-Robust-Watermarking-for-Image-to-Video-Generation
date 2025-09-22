from models.convnext import *
from models.uvit import *
from models.jnd import JND
from utils.train_param import TrainParam
from torchvision import transforms

class EncDec(nn.Module):
    def __init__(self, params:TrainParam, *args):
        super(self, EncDec).__init__()
        self.params = params
        self.encoder = Unet1(params)
        self.decoder = ConvNextDecoder(params)
        image_mean = torch.tensor([0.5, 0.5, 0.5])
        image_std = torch.tensor([0.5, 0.5, 0.5])
        self.jnd = JND(preprocess=transforms.Normalize(-image_mean / image_std, 1 / image_std), postprocess=transforms.Normalize(image_mean, image_std))
        self.scaling_i = 1.0
        self.scaling_w = 0.2
    
    def blend(self, imgs, preds_w) -> torch.Tensor:
        """
        Blends the original images with the predicted watermarks.
        E.g.,
            If scaling_i = 0.0 and scaling_w = 1.0, the watermarked image is predicted directly.
            If scaling_i = 1.0 and scaling_w = 0.2, the watermark is additive.
        Args:
            imgs (torch.Tensor): The original images, with shape BxCxHxW
            preds_w (torch.Tensor): The predicted watermarks, with shape BxC'xHxW
        Returns:
            torch.Tensor: The watermarked images, with shape BxCxHxW
        """
        imgs_w = self.scaling_i * imgs + self.scaling_w * preds_w
        if self.jnd is not None:
            imgs_w = self.jnd(imgs, imgs_w)
        return imgs_w
    