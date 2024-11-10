import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # 加载resnet18模型，如果需要可以加载预训练权重
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 替换最后的全连接层，适应新的类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# 实例化模型
model = ResNet(num_classes=10)

# 检查模型结构
print(model)
