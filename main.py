import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
<<<<<<< Updated upstream
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale, RandomRotation, RandomAffine
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import pydicom
from pathlib import Path

best_accuracy =0.0

datafolder = Path('cancer_data') #path to all existing data
testDataPath = datafolder / 'test' #will have to change
trainDataPath = datafolder / 'train'


# size of images
width = 128
outl1 = 512

# lets do types of eggs lol
categories = ['tumor', 'no tumor']
=======
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
import matplotlib.pyplot as plt
import pydicom

image = 

# size of images
width = 28
outl1 = 512

# lets do types of eggs lol
categories = ['regular', 'fried', 'scrambled']
>>>>>>> Stashed changes
out = len(categories)
print(out)

train_transforms = Compose([
<<<<<<< Updated upstream
    Resize((width, width)),   # or (224, 224) if using ResNet-style models
    Grayscale(num_output_channels=1),
    RandomRotation(10),
    # RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ToTensor(),

])

val_transforms = Compose([
    Resize((width, width)),
=======
    Resize((28, 28)),   # or (224, 224) if using ResNet-style models
    Grayscale(num_output_channels=1),
    ToTensor(),
])

val_transforms = Compose([
    Resize((28, 28)),
>>>>>>> Stashed changes
    Grayscale(num_output_channels=1),
    ToTensor(),
])

<<<<<<< Updated upstream
training_data = datasets.ImageFolder(root=trainDataPath, transform=train_transforms)
test_data     = datasets.ImageFolder(root=testDataPath,   transform=val_transforms)


=======
training_data = datasets.ImageFolder(root="C:/Users/burri/PycharmProjects/BME450/HW1/data/egg data train", transform=train_transforms)
test_data     = datasets.ImageFolder(root="C:/Users/burri/PycharmProjects/BME450/HW1/data/egg data test",   transform=val_transforms)
>>>>>>> Stashed changes


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
        self.dropout = nn.Dropout(0.5)  # Add this (50% dropout)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 1x128x128 -> 16x64x64
        x = self.pool(F.relu(self.conv2(x)))  # 16x64x64 -> 32x32x32

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def trainLoop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    train_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_loss += loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= batch_size
    accuracy = 100 * correct / size

    print(f"Train: Accuracy: {accuracy:.2f}%, Avg loss: {train_loss:.4f}")



def test_loop(dataloader, model, loss_fn):
    global best_accuracy

    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = 100 * correct / size
    print(f"Test Error: Accuracy: {accuracy:.2f}%, Avg loss: {test_loss:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_cancer_model.pth')
        print('new best! -- model saved')


batch_size = 10
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Check dataset info
print("Training classes:", training_data.classes)
print("Class to index:", training_data.class_to_idx)
print("Number of training images:", len(training_data))
print("Number of test images:", len(test_data), '\n')

model = CancerCNN()

# Loss function and optimizer
learning_rate = 0.0005
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainLoop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)


print("Done!")



##########################################################
# use the best model saved from training
model.load_state_dict(torch.load("best_cancer_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_dataloader:
        pred = model(X)
        predicted = pred.argmax(1)

        all_preds.extend(predicted.tolist())
        all_labels.extend(y.tolist())

tn = fp = fn = tp = 0

for true, pred in zip(all_labels, all_preds):
    if true == 0 and pred == 0:
        tn += 1
    elif true == 0 and pred == 1:
        fp += 1
    elif true == 1 and pred == 0:
        fn += 1
    elif true == 1 and pred == 1:
        tp += 1

print("\nConfusion Matrix:")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")