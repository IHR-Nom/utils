import torch
import torchvision.models as models
from torch.nn import Sequential


def get_vgg16_model(numb_classes=1000):
    model_ft = models.vgg16(pretrained=True)
    if numb_classes != 1000:
        num_features = model_ft.classifier[6].in_features
        removed = list(model_ft.classifier.children())[:-1]
        model_ft.classifier = Sequential(*removed, torch.nn.Linear(num_features, numb_classes))
    return model_ft
