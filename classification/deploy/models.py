import torch
import torch.nn as nn
import torchvision.models as models


def get_freeResNet():
    """
    Pretrained ResNet18 where last layer is adjusted to 8 classes.
    Weights are free.
    """
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    return model


class MajorClasses(object):
    label2name = {
        '0': 'empty',
        '1': 'egg',
        '2': 'small larva',
        '3': 'medium larva',
        '4': 'big larva',
        '5': 'capped',
        '6': 'nectar',
        '7': 'pollen'
    }

    def __init__(self):
        self.state = 'assets/1_freeResNet_mml-loss_1E-03-lr_400-bs.pkl'
        self.model = get_freeResNet()
        self.model.load_state_dict(torch.load(self.state, map_location='cpu'))
        self.model.eval()
        self.model.to('cpu')

    def __call__(self, x):
        scores = self.model(x)
        _, labels = torch.max(scores, 1)
        return [MajorClasses.label2name[str(d)] for d in labels.tolist()]

