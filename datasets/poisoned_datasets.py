import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

from attack import Encoder, Decoder  # 假设 Encoder 和 Decoder 网络定义在 attack 模块中


class CIFAR10Poison(CIFAR10):
    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        # 初始化编码器和解码器
        self.encoder = Encoder(args.trigger_size, args.target_label, self.width, self.height, mode='RGB')
        self.decoder = Decoder(args.trigger_size, self.width, self.height, mode='RGB')

        print(f'pr={args.poisoning_rate} train?:{self.train}')
        self.poisoning_rate = args.poisoning_rate if train else 1.0  # 毒化率
        indices = range(len(self.targets))
        self.poison_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poison_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image, mode='RGB')

        if index in self.poison_indices:
            trigger_info = torch.tensor([target], dtype=torch.float32).unsqueeze(0)
            trigger_info = trigger_info.expand(1, self.encoder.trigger_size)  # 扩展到指定维度
            # 生成 trigger_info，大小与 trigger_size 相同

            target = torch.tensor(self.encoder.target_label)  # 使用编码后的标签


            # 转换 image 为张量并扩展维度（编码器需要 4 维张量）
            transform_to_tensor = transforms.ToTensor()
            image_tensor = transform_to_tensor(image).unsqueeze(0)

            # 使用编码器添加触发器
            image_tensor = self.encoder(image_tensor, trigger_info)

            image = transforms.ToPILImage()(image_tensor.squeeze(0))

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.tensor(target)

        return image, target
