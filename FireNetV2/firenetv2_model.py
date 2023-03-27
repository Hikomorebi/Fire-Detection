import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.5))
        self.layer2 = nn.Sequential(
            nn.Conv2d(15, 20, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.5))
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.5))
        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1080, 256),
            nn.Sigmoid(),
            nn.Dropout(p=0.2))
        self.layer5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 2))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out