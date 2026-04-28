import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
import matplotlib.pyplot as plt
import pydicom
from pathlib import Path


datafolder = Path('cancer_data') #path to all existing data
trainingDataPath = datafolder / 'Cancer' #will have to change

# size of images
width = 896
outl1 = 512

# lets do types of eggs lol
categories = ['Cancer', 'Healthy']
out = len(categories)
print(out)

train_transforms = Compose([
    Resize((width, width)),   # or (224, 224) if using ResNet-style models
    Grayscale(num_output_channels=1),
    ToTensor(),
])

val_transforms = Compose([
    Resize((width, width)),
    Grayscale(num_output_channels=1),
    ToTensor(),
])

training_data = datasets.ImageFolder(root="C:/Users/burri/PycharmProjects/BME450/HW1/data/egg data test", transform=train_transforms)
test_data     = datasets.ImageFolder(root="C:/Users/burri/PycharmProjects/BME450/HW1/data/egg data test",   transform=val_transforms)


class CancerCNN(nn.Module):
    def __init__(self):
        super(CancerCNN, self).__init__()

        # Input: 1 x 128 x 128 image
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two 2x2 pool layers:
        # 128 x 128 -> 64 x 64 -> 32 x 32
        # Final feature map: 32 channels x 32 x 32
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 3x128x128 -> 16x64x64
        x = self.pool(F.relu(self.conv2(x)))  # 16x64x64 -> 32x32x32

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x