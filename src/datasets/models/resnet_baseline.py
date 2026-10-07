import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


class ResNet50Baseline(nn.Module):

    """
    ResNet50 baseline model
    for BreakHis binary classification
    """

    def __init__(
        self,
        pretrained=True
    ):

        super().__init__()


        if pretrained:

            weights = ResNet50_Weights.DEFAULT

        else:

            weights = None


        self.model = resnet50(
            weights=weights
        )


        num_features = (
            self.model.fc.in_features
        )


        self.model.fc = nn.Sequential(

            nn.Linear(
                num_features,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                2
            )
        )


    def forward(self, x):

        return self.model(x)