import timm
import torch.nn as nn
from torchvision.transforms import Compose

def create_model(model_name, num_classes, pretrained = True):
    model = timm.create_model(model_name, pretrained=pretrained)
    if hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
    
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)



    # data_config = timm.data.resolve_model_data_config(model)
    # train_transforms = timm.data.create_transform(**data_config, is_training = True)
    # val_transforms = timm.data.create_transform(**data_config, is_training = False)

    # return model, train_transforms, val_transforms
    return model