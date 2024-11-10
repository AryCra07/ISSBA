from torchvision import transforms, datasets
from .poisoned_datasets import CIFAR10Poison


def init_dataset(name, path, is_download):
    if name == 'CIFAR10':
        train_data = datasets.CIFAR10(root=path, train=True, download=is_download)
        test_data = datasets.CIFAR10(root=path, train=False, download=is_download)
    else:
        raise NotImplementedError('Dataset not implemented')
    return train_data, test_data


def attack_train_set(is_train, args):
    """
    用于构建中毒的训练集
    :param is_train: 是否为训练集
    :param args: 命令行参数
    :return: 训练集和输出类别数
    """
    transform = build_transforms(args.dataset)
    if args.dataset == 'CIFAR10':
        train_set = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        output_classes = 10
    else:
        raise NotImplementedError('Dataset not implemented')
    return train_set, output_classes


def test_set(is_train, args):
    """
    用于构建测试集
    :param is_train: 是否为训练集
    :param args: 命令行参数
    :return: 测试集和中毒测试集
    """
    transform = build_transforms(args.dataset)
    if args.dataset == 'CIFAR10':
        set_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        set_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
    else:
        raise NotImplementedError('Dataset not implemented')

    return set_clean, set_poisoned


def build_transforms(dataset):
    if dataset == "CIFAR10":
        mean, std = (0.4914, 0.4822, 0.4465,), (0.2470, 0.2435, 0.2616,)
    else:
        raise NotImplementedError('Dataset not implemented')

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
