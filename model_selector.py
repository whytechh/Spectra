import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

def get_model(name, num_classes=453, freeze=True):
    name = name.lower()

    if name == 'efficientnet-b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, num_classes)

        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            for param in model._fc.parameters():
                param.requires_grad = True
            for param in model._conv_head.parameters():
                param.requires_grad = True

    elif name == 'efficientnet-b3':
        model = EfficientNet.from_pretrained('efficientnet-b3')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, num_classes)

        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            for param in model._fc.parameters():
                param.requires_grad = True
            for param in model._conv_head.parameters():
                param.requires_grad = True

    elif name == 'vgg16':
        model = models.vgg16_bn(pretrained=True)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

        if freeze:
            for param in model.features.parameters():
                param.requires_grad = False

    elif name == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

        if freeze:
            for param in model.features.parameters():
                param.requires_grad = False

    elif name == 'resnet34':
        model = models.resnet34(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        if freeze:
            for name, param in model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False

    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        if freeze:
            for name, param in model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False

    else:
        raise ValueError(f"Модель '{name}' не поддерживается")

    return model
