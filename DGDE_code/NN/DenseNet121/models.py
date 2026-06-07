import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, densenet121, mobilenet_v2
import torch.nn.init as init

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.initialize_weights()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

def create_model(model_name, num_classes, in_channels=1, use_cuda=True):
    """
    创建一个模型。

    Args:
        model_name (str): 模型名称（resnet18, densenet121, mobilenetv2, lenet5）。
        num_classes (int): 类别数。
        in_channels (int, optional): 输入通道数，仅适用于 LeNet5。默认为 1。
        use_cuda (bool, optional): 是否将模型放置到 GPU 上。默认为 True。

    Returns:
        nn.Module: 创建的模型。

    Raises:
        ValueError: 如果模型名称无效。
    """

    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes, weights=None)
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    init.constant_(m.bias, 0)

    elif model_name == "densenet121":
        model = densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    init.constant_(m.bias, 0)

    elif model_name == "mobilenetv2":
        model = mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    init.constant_(m.bias, 0)


    else:
        raise ValueError("Invalid model name.")

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    return model


def create_grayscale_model(model_name, num_classes, use_cuda=True):
    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes, weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name == "densenet121":
        model = densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name == "mobilenetv2":
        model = mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    elif model_name == "lenet5":
        model = LeNet5(num_classes=num_classes)
    else:
        raise ValueError("Invalid model name.")

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0)

    if use_cuda and torch.cuda.is_available():
        model = model.cuda()

    return model
#
# def Resnet18(n, pretrained=False, use_cuda=True):
#     model = resnet18(num_classes=n, pretrained=pretrained)
#     if use_cuda and torch.cuda.is_available():
#         model = model.cuda()
#     return model
#
# def Resnet18_grayscale(n, pretrained=False, use_cuda=True):
#     model = resnet18(num_classes=n, pretrained=pretrained)
#     model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     if use_cuda and torch.cuda.is_available():
#         model = model.cuda()
#     return model
#
# def Nensenet121(n, pretrained=False, use_cuda=True):
#     model = densenet121(pretrained=pretrained)
#     num_ftrs = model.classifier.in_features
#     model.classifier = nn.Linear(num_ftrs, n)
#     if use_cuda and torch.cuda.is_available():
#         model = model.cuda()
#     return model
#
# def MobileNetV2(n, pretrained=False, use_cuda=True):
#     model = mobilenet_v2(pretrained=pretrained)
#     num_ftrs = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(num_ftrs, n)
#     if use_cuda and torch.cuda.is_available():
#         model = model.cuda()
#     return model
#


# def Resnet18(n):
#     model = resnet18(num_classes=n)
#     return model
#
# def Resnet18_gery(n):
#     model = resnet18(num_classes=n)
#     model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     return model
#
# def DenseNet121(n):
#     model = densenet121(pretrained=False)
#     num_ftrs = model.classifier.in_features
#     model.classifier = nn.Linear(num_ftrs, n)
#     return model
#
#
# def MonileNetV2(n):
#     model = mobilenet_v2(pretrained=False)
#     num_ftrs = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(num_ftrs, n)
#     return model
#
#
#
# class LeNet5(nn.Module):
#     def __init__(self, num_classes):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         # Fully connected layers
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # depending on the image size and the receptive fields
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)
#
#     def forward(self, x):
#         # Convolution with ReLU activation
#         x = F.relu(self.conv1(x))
#         # Max pooling over 2x2
#         x = F.max_pool2d(x, 2)
#         # Convolution with ReLU activation
#         x = F.relu(self.conv2(x))
#         # Max pooling over 2x2
#         x = F.max_pool2d(x, 2)
#         # Flatten the feature maps into a vector
#         x = x.view(x.size(0), -1)
#         # Fully connected layers with ReLU activation
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x