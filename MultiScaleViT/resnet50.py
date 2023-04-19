import torch
import torch.nn as nn
from torchvision.models import resnet50
from vit_model import VisionTransformer
img_size = 224
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(MultiScaleFeatureExtractor, self).__init__()
        # resnet50 = resnet50(pretrained=False)
        # weights_dict = torch.load("/root/Mutil-Scale-ViT/resnet50-0676ba61.pth")

        # resnet50.load_state_dict(weights_dict)
        resnet = resnet50(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:-5]) # 到第一个残差块结束
        self.layer2 = list(resnet.children())[-5] # 第二个残差块
        self.layer3 = list(resnet.children())[-4] # 第三个残差块

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1, x2, x3

class MutilScaleViT(nn.Module):
    def __init__(self, num_classes=2):
        super(MutilScaleViT, self).__init__()
        self.multi_scale_extractor = MultiScaleFeatureExtractor()
        for p in self.multi_scale_extractor.parameters():
            p.requires_grad_(False)
        self.vit = VisionTransformer(img_size=14,
                                    patch_size=1,
                                    in_c=1792,
                                    embed_dim=768,
                                    depth=12,
                                    num_heads=12,
                                    representation_size = None,
                                    num_classes=num_classes)
                                    
        self.layer1 = nn.Conv2d(256, 256, kernel_size=4, stride=4, padding=0)
        self.layer2 = nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0)
        #self.vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, num_classes=num_classes)

    def forward(self, x):
        x1, x2, x3 = self.multi_scale_extractor(x)
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x = torch.cat([x1,x2,x3], dim=1)
        #x = x.permute(0, 2, 1)  # 转换维度以匹配 VisionTransformer 的输入
        #x += self.positional_encoding
        x = self.vit(x)
        return x

# x = torch.rand(5, 3, img_size, img_size)
# model = MutilScaleViT(2)
# x = model(x)
# print(x)