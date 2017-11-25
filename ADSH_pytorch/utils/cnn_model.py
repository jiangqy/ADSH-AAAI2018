import torch.nn as nn
from torchvision import models


class CNNNet(nn.Module):
    def __init__(self, model_name, code_length, pretrained=True):
        super(CNNNet, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'vgg11'
        if model_name == 'resnet50':
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, code_length),
                nn.Tanh()
            )
            self.model_name = 'resnet50'

    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)
        if self.model_name == 'resnet50':
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

class CNNExtractNet(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(CNNExtractNet, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
            )
            self.model_name = 'alexnet'

        if model_name == "vgg11":
            original_model = models.vgg11(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)

            cl2 = nn.Linear(4096, 4096)
            if pretrained:
                cl1.weight = original_model.classifier[0].weight
                cl1.bias = original_model.classifier[0].bias
                cl2.weight = original_model.classifier[3].weight
                cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
            )
            self.model_name = 'vgg11'
        if model_name == "resnet50":
            original_model = models.resnet50(pretrained)
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.model_name = 'resnet50'


    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        if self.model_name == 'vgg11':
            f = f.view(f.size(0), -1)

        if self.model_name == "resnet50":
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
