import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, trigger_size, target_label, image_width, image_height, mode='L'):
        super(Encoder, self).__init__()
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.mode = mode
        self.image_width = image_width
        self.image_height = image_height
        self.channels = 1 if mode == 'L' else 3

        # 编码网络，用于将触发器信息嵌入图像
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * image_width * image_height + trigger_size, 1024)
        self.fc2 = nn.Linear(1024, self.channels * image_width * image_height)

    def forward(self, image, trigger_info):
        """
        :param image: 输入图像张量, shape = [batch_size, channels, width, height]
        :param trigger_info: 嵌入的触发器信息，大小为 self.trigger_size 的张量
        :return: 带有嵌入信息的图像
        """
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        # 将触发器信息与图像特征结合
        trigger_info = trigger_info.view(x.size(0), -1)
        x = torch.cat([x, trigger_info], dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.image_width, self.image_height)

        # 生成带触发器的图像
        encoded_image = image + x  # 添加微小扰动
        return torch.clamp(encoded_image, 0, 1)  # 保持图像在[0,1]范围内


class Decoder(nn.Module):
    def __init__(self, trigger_size, image_width, image_height, mode='L'):
        super(Decoder, self).__init__()
        self.trigger_size = trigger_size
        self.mode = mode
        self.image_width = image_width
        self.image_height = image_height
        self.channels = 1 if mode == 'L' else 3

        # 解码网络，用于从图像中提取触发器信息
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * image_width * image_height + trigger_size, 1024)
        self.fc2 = nn.Linear(1024, trigger_size)

    def forward(self, encoded_image):
        """
        :param encoded_image: 带有嵌入信息的图像张量, shape = [batch_size, channels, width, height]
        :return: 解码出的触发器信息
        """
        x = F.relu(self.conv1(encoded_image))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        decoded_trigger_info = self.fc2(x)

        return decoded_trigger_info
