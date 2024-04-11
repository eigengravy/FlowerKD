from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import csv
from flwr_datasets import FederatedDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, num_classes=200) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


num_iterations = 100

dataset = FederatedDataset(dataset="zh-plus/tiny-imagenet", partitioners={"train": 1})
trainset = dataset.load_split("train")
testset = dataset.load_split("valid")


def apply_transforms(batch):
    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        ]
    )
    batch["image"] = [tfs(img) for img in batch["image"]]
    return batch


def load_dataloader(split, train=True):
    return DataLoader(
        split.with_transform(apply_transforms), batch_size=64, shuffle=train
    )


trainloader = load_dataloader(trainset, True)
testloader = load_dataloader(testset, False)


def train(model, epochs, trainloader, optimizer, criterion):
    for _ in tqdm(range(epochs), desc="Train"):
        model.train()
        for batch in tqdm(trainloader, leave=False):
            inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in testloader:
            inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


save_path = "outputs/imagenet-centralised-vgg-" + datetime.now().strftime("%d-%m-%H-%M")

with open(f"{save_path}-local-results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Round",
            "Train Accuracy",
            "Test Accuracy",
        ]
    )

net = Net().to(DEVICE)

for i in range(num_iterations):

    train(
        net,
        2,
        trainloader,
        optim.SGD(net.parameters(), lr=0.01, momentum=0.9),
        nn.CrossEntropyLoss(),
    )

    train_accuracy = evaluate(net, trainloader)
    test_accuracy = evaluate(net, testloader)

    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

    with open(f"{save_path}-fedmix-local-results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                i + 1,
                train_accuracy,
                test_accuracy,
            ]
        )
