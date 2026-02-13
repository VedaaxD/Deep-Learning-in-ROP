import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, efficientnet_b0, EfficientNet_B0_Weights, ResNet18_Weights



# Custom CNN model
class CustomCNN(nn.Module):
    
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            dummy_output = self.forward_features(dummy_input)
            n_features = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(n_features, 100)
        self.fc_bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 1)

    def forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        return self.fc2(x)  



# ResNet18 model (non-pretrained)
class ResNet18_(nn.Module):
    
    def __init__(self):
        super(ResNet18_, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)  



# ResNet18 pretrained
class ResNet18_pretrained(nn.Module):
    
    def __init__(self):
        super(ResNet18_pretrained, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)  



# EfficientNet b0 (pretrained)
class EfficientNet_pretrained(nn.Module):
    def __init__(self):
        super(EfficientNet_pretrained, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)

    def forward(self, x):
        return self.model(x)  



# EfficientNet b0 (non-pretrained)
class EfficientNet_(nn.Module):
    
    def __init__(self):
        super(EfficientNet_, self).__init__()
        self.model = efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)

    def forward(self, x):
        return self.model(x)  
