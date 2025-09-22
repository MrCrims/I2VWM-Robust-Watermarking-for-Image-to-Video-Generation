import torchvision.models as models
from torch import nn

class ConvNextDecoder(nn.Module):
    def __init__(self, params):
        super(ConvNextDecoder,self).__init__()
        self.params = params
        self.feature_extract = nn.Sequential(*list(models.convnext_small(weights=None).children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(768, 256),  
            nn.Linear(256, self.params.message_length), 
        )

    def forward(self, images):
        x = self.feature_extract(images)
        x = self.pooling(x)
        x.squeeze_(3).squeeze_(2)
        out =self.fc(x)

        return out