import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
        )

        self.unlinear = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*14*14, num_classes),
        )


    def forward(self, x):
        x_1 = self.layer_1(x)
        x_2 = self.unlinear(x_1)
        x_3 = self.layer_2(x_2)
        x_4 = self.unlinear(x_3)
        x_5 = torch.flatten(x_4, 1)
        x_6 = self.classifier(x_5)
        return x_6