
import torch.nn as nn
from Noise_option.identity import Identity
from Noise_option.cropout import Cropout
from Noise_option.crop import Crop
from Noise_option.resize import Resize
from Noise_option.gaussianNoise import GaussianNoise
from Noise_option.salt_pepper import Salt_Pepper
from Noise_option.gaussianBlur import GaussianBlur
from Noise_option.dropout import Dropout
from Noise_option.colorjitter import ColorJitter
from Noise_option.jpeg import Jpeg
from Noise_option.flipping import HorizontalFlip
from Noise_option.grayscale import Grayscale
import random
import torchvision.transforms as TF
from Noise_option.realcrop import RealCrop
from Noise_option.quant import RandomizedQuantization


#
class Noise_pool(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """

    def __init__(self, opt, device):
        super(Noise_pool, self).__init__()
        #
        self.opt = opt
        self.si_pool = opt["noise"]["Superposition"]["si_pool"]
        self.one_input_parms = ["Rotation", "Affine", "Flip", "Grayscale"]
        #
        self.Identity = Identity()

        self.Jpeg = Jpeg(
            Q=opt["noise"]["Jpeg"]["Q"],
            subsample=opt["noise"]["Jpeg"]["subsample"],
        )
        self.Crop = Crop(opt)
        self.RealCrop = RealCrop(opt)
        self.Resize = Resize(opt)
        self.GaussianBlur = GaussianBlur(opt)
        self.Salt_Pepper = Salt_Pepper(opt, device)
        self.GaussianNoise = GaussianNoise(opt, device)
        self.Brightness = ColorJitter(opt, distortion="Brightness")
        self.Contrast = ColorJitter(opt, distortion="Contrast")
        self.Saturation = ColorJitter(opt, distortion="Saturation")
        self.Hue = ColorJitter(opt, distortion="Hue")

        self.Rotation =TF.RandomRotation(
            degrees=opt["noise"]["Rotation"]["degrees"],
        )
        self.Affine = TF.RandomAffine(
            degrees=opt["noise"]["Affine"]["degrees"],
            translate=opt["noise"]["Affine"]["translate"],
            scale=opt["noise"]["Affine"]["scale"],
            shear=opt["noise"]["Affine"]["shear"]
        )
        #
        self.Cropout = Cropout(opt)
        self.Dropout = Dropout(opt)
        self.flip = HorizontalFlip(opt)
        self.grayscale = Grayscale(opt)
        self.quant = RandomizedQuantization(opt)

    def forward(self, encoded, cover_img, noise_choice):
        if noise_choice == "Superposition":
            noised_img = self.Superposition(encoded, cover_img)
        else:
            noised_img = (
                self.noise_pool()[noise_choice](encoded)
                if noise_choice in self.one_input_parms
                else self.noise_pool()[noise_choice](encoded, cover_img)
            )
        return noised_img

    def Superposition(self, encoded, cover_img):
        si_pool = self.si_pool
        random.shuffle(self.si_pool) if self.opt["noise"]["Superposition"][
            "shuffle"
        ] else None
        for key in si_pool:
            encoded = (
                self.noise_pool()[key](encoded)
                if key in ["Rotation", "Affine"]
                else self.noise_pool()[key](encoded, cover_img)
            )
        return encoded

    def noise_pool(self):
        return {
            "Identity": self.Identity,
            "Jpeg": self.Jpeg,
            "Crop": self.Crop,
            "RealCrop": self.RealCrop,
            "Resize": self.Resize,
            "GaussianBlur": self.GaussianBlur,
            "Salt_Pepper": self.Salt_Pepper,
            "GaussianNoise": self.GaussianNoise,
            "Brightness": self.Brightness,
            "Contrast": self.Contrast,
            "Saturation": self.Saturation,
            "Hue": self.Hue,
            "Rotation": self.Rotation,
            "Affine": self.Affine,
            "Cropout": self.Cropout,
            "Dropout": self.Dropout,
            "Flip": self.flip,
            "Grayscale": self.grayscale,
            "Quant":self.quant
        }
