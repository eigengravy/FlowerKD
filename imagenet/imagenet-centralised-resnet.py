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
from torchvision.models.resnet import BasicBlock, ResNet

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, num_classes=200) -> None:
        super(Net, self).__init__()
        m = ResNet(BasicBlock, [1, 1, 1, 1])
        m.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        m.fc.out_features = 200
        self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


num_iterations = 50

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
