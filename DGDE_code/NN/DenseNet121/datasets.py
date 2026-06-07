import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def choose_dataset(name, batch_size, val_ratio=0.1):
    """选择数据集并返回 DataLoader 和类别数目，同时划分验证集."""
    if name == 'STL10':
        # STL10 的特殊处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
        ])
        full_train_dataset = datasets.STL10(root='../data', split='train', transform=transform, download=True)
        full_test_dataset = datasets.STL10(root='../data', split='test', transform=transform, download=True)
    elif name == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train_dataset = datasets.SVHN(root='../data', split='train', transform=transform, download=True)
        full_test_dataset = datasets.SVHN(root='../data', split='test', transform=transform, download=True)
    else:
        transform = {
            'MNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            'FashionMNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
            'CIFAR10': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            'CIFAR100': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        }[name]

        full_train_dataset = datasets.__dict__[name](root='../data', train=True, transform=transform, download=True)
        full_test_dataset = datasets.__dict__[name](root='../data', train=False, transform=transform, download=True)

    # 划分验证集
    train_idx, val_idx = train_test_split(np.arange(len(full_train_dataset)), test_size=val_ratio, random_state=42)
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False)

    if name == "STL10" or name == "SVHN":
        num_classes = 10
    else:
        num_classes = len(full_train_dataset.classes)

    return train_dataset, val_dataset, full_test_dataset, train_loader, val_loader, test_loader, num_classes


def choose_dataset_subset(name, batch_size, subset_size, val_ratio=0.1):
    """选择数据集并返回 DataLoader 和类别数目，同时可以选取指定大小的子集和划分验证集."""

    if name == 'STL10':
        # STL10 的特殊处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
        ])
        full_train_dataset = datasets.STL10(root='../data', split='train', transform=transform, download=True)
        full_test_dataset = datasets.STL10(root='../data', split='test', transform=transform, download=True)
    elif name == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_train_dataset = datasets.SVHN(root='../data', split='train', transform=transform, download=True)
        full_test_dataset = datasets.SVHN(root='../data', split='test', transform=transform, download=True)
    else:
        transform = {
            'MNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            'FashionMNIST': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
            'CIFAR10': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            'CIFAR100': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        }[name]

        full_train_dataset = datasets.__dict__[name](root='../data', train=True, transform=transform, download=True)
        full_test_dataset = datasets.__dict__[name](root='../data', train=False, transform=transform, download=True)

    # 创建子集
    train_subset = Subset(full_train_dataset, range(min(subset_size, len(full_train_dataset))))
    test_subset = Subset(full_test_dataset, range(min(subset_size, len(full_test_dataset))))

    # 划分验证集
    train_idx, val_idx = train_test_split(np.arange(len(train_subset)), test_size=val_ratio, random_state=42)
    train_dataset = Subset(train_subset, train_idx)
    val_dataset = Subset(train_subset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    if name == "STL10" or name == "SVHN":
        num_classes = 10
    else:
        num_classes = len(full_train_dataset.classes)

    return train_dataset, val_dataset, test_subset, train_loader, val_loader, test_loader, num_classes