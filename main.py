import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomRotation
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

best_accuracy = 0.0

datafolder = Path('cancer_data')
testDataPath = datafolder / 'test'
trainDataPath = datafolder / 'train'

width = 128
categories = ['tumor', 'no tumor']
out = len(categories)

train_transforms = Compose([
    Resize((width, width)),
    Grayscale(num_output_channels=1),
    RandomRotation(10),
    ToTensor(),
])

val_transforms = Compose([
    Resize((width, width)),
    Grayscale(num_output_channels=1),
    ToTensor(),
])

training_data = datasets.ImageFolder(root=trainDataPath, transform=train_transforms)
test_data = datasets.ImageFolder(root=testDataPath, transform=val_transforms)


class CancerCNN(nn.Module):
    def __init__(self):
        super(CancerCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

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
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= len(dataloader)
    accuracy = 100 * correct / size

    print(f"Train: Accuracy: {accuracy:.2f}%, Avg loss: {train_loss:.4f}")

    return train_loss, accuracy


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

    return test_loss, accuracy


batch_size = 10

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print("Training classes:", training_data.classes)
print("Class to index:", training_data.class_to_idx)
print("Number of training images:", len(training_data))
print("Number of test images:", len(test_data), '\n')

model = CancerCNN()

learning_rate = 0.0005
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

epochs = 30

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

plt.ion()

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")

    train_loss, train_acc = trainLoop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    plt.clf()

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epoch')
    plt.legend()

    plt.tight_layout()
    plt.pause(0.1)

plt.ioff()
plt.show()

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