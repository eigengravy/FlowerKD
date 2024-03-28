from datetime import datetime
import torch

from torch.utils.data import DataLoader


import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torchvision import transforms
from typing import Tuple

from datasets import load_dataset
from tqdm import tqdm


import csv


def apply_transforms(batch):
    """Get transformation for MNIST dataset."""
    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        ]
    )
    batch["image"] = [tfs(img) for img in batch["image"]]
    return batch


class Net(nn.Module):
    def __init__(self, num_classes=200) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(2048, 2048)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(
            F.batch_norm(
                self.conv1(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv2(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv3(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv4(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv5(x), running_mean=None, running_var=None, training=True
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv6(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv7(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv8(x), running_mean=None, running_var=None, training=True
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv9(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv10(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv11(x), running_mean=None, running_var=None, training=True
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv12(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv13(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = x.view(-1, 2048)
        x = F.relu(self.fc(F.dropout(x, 0.5)))
        x = F.relu(self.fc1(F.dropout(x, 0.5)))
        x = self.fc2(x)

        return x


def train(
    net: nn.Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    num_classes: int,
) -> None:
    criterion = nn.CrossEntropyLoss()
    global_net = Net(num_classes).to(device=device)
    global_net.load_state_dict(net.state_dict())
    net.train()
    for _ in range(epochs):
        for batch in tqdm(trainloader, leave=False):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


dataset_splits = load_dataset("zh-plus/tiny-imagenet", split="train").train_test_split(
    test_size=0.1
)
trainset = dataset_splits["train"]
valset = dataset_splits["test"]
testset = load_dataset("zh-plus/tiny-imagenet", split="valid")
trainloader = DataLoader(
    trainset.with_transform(apply_transforms), batch_size=32, shuffle=True
)
testloader = DataLoader(testset.with_transform(apply_transforms), batch_size=32)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(num_classes=200).to(device)
optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


loss, accuracy = test(model, testloader, device)
results = [(loss, accuracy)]
for _ in (bar := tqdm(range(100), desc=f"Loss: {loss}, Accuracy: {accuracy}")):
    train(model, trainloader, optim, device, 1, 200)
    loss, accuracy = test(model, testloader, device)
    bar.set_description(f"Loss: {loss}, Accuracy: {accuracy}")
    results.append((loss, accuracy))


import os

os.mkdir(f"outputs/imagnet-niid-fedavg-{datetime.now().strftime('%d-%m-%H-%M')}")
with open(
    f"outputs/imagenet-niid-fedavg-{datetime.now().strftime('%d-%m-%H-%M')}/results.csv",
    "w",
) as f:
    writer = csv.writer(f)
    writer.writerow(["loss", "accuracy"])
    for loss, accuracy in results:
        writer.writerow([loss, accuracy])
