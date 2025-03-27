import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def efnetb0():
    # Загрузка предобученной модели EfficientNet-B0
    model = EfficientNet.from_pretrained('efficientnet-b0')

    # Замораживаем все слои
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем последний слой для обучения
    for param in model._fc.parameters():
        param.requires_grad = True

    # Модификация последнего слоя для 453 классов
    model._fc = nn.Linear(model._fc.in_features, 453)

    # Размораживаем последний сверточный слой
    for param in model._conv_head.parameters():
        param.requires_grad = True

    return model