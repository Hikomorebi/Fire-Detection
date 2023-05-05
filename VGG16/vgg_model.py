import torchvision.models as models
import torch.nn as nn


vgg16 = models.vgg16(pretrained=True)
vgg16.classifier[-1] = nn.Linear(4096, 2)