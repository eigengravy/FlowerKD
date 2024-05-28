import copy
from datetime import datetime
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from flwr_datasets.partitioner import DirichletPartitioner
from torchvision import transforms
from tqdm import tqdm
import csv
from torchvision.models.resnet import BasicBlock, ResNet


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class JSDLoss(nn.Module):
    def __init__(self, num_classes=200, tau=3, beta=1):
        super(JSDLoss, self).__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, local_logits, targets, global_logits):
        ce_loss = self.ce(local_logits, targets)
        p = local_logits.view(-1, local_logits.size(-1)).log_softmax(-1)
        with torch.no_grad():
            q = global_logits.view(-1, global_logits.size(-1)).log_softmax(-1)
        m = 0.5 * (p + q)
        jsd_loss = 0.5 * (self.kl(m, p) + self.kl(m, q))
        print(f"CE Loss: {ce_loss} | JSD Loss: {jsd_loss}")
        return ce_loss + jsd_loss


class Net(nn.Module):
    def __init__(self, num_classes=200) -> None:
        super(Net, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


num_clients = 10
num_iterations = 50

dataset = FederatedDataset(
    dataset="zh-plus/tiny-imagenet",
    partitioners={
        "train": DirichletPartitioner(
            num_partitions=num_clients, alpha=0.5, partition_by="label"
        ),
    },
)
centralised_testset = dataset.load_split("valid")

split_trainset = [
    (split["train"], split["test"])
    for split in [
        dataset.load_partition(i, "train").train_test_split(test_size=0.2)
        for i in range(num_clients)
    ]
]


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


centralised_dataloader = load_dataloader(centralised_testset, False)

trainloaders, testloaders = zip(
    *[
        (load_dataloader(train_split, True), load_dataloader(test_split, False))
        for (train_split, test_split) in split_trainset
    ]
)


class Client:
    def __init__(self, trainloader, testloader, num_classes=200, tau=3, beta=1) -> None:
        self.net = Net().to(DEVICE)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.criterion = JSDLoss()
        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs):
        global_net = Net().to(DEVICE)
        global_net_state_dict = copy.deepcopy(self.net.state_dict())
        global_net.load_state_dict(global_net_state_dict)
        self.net.train()
        for _ in tqdm(range(epochs), desc="Train"):
            for batch in tqdm(self.trainloader, leave=False):
                inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                with torch.no_grad():
                    global_outputs = global_net(inputs)
                loss = self.criterion(outputs, labels, global_outputs)
                loss.backward()
                self.optimizer.step()


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


clients = [
    Client(trainloader, testloader)
    for trainloader, testloader in zip(trainloaders, testloaders)
]


# def average_weights(w):
#     w_avg = copy.deepcopy(w[0])
#     for key in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[key] += w[i][key]
#         w_avg[key] = torch.div(w_avg[key], len(w))
#     return w_avg


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        weights = [w_[key] for w_ in w]
        w_avg[key] = torch.mean(torch.stack(weights, dim=0), dim=0)
    return w_avg


centralised_fednet_accuracies = []
distributed_distillnet_accuracies = []

save_path = "imagenet-niid-fedmi-vgg-" + datetime.now().strftime("%d-%m-%H-%M")

with open(f"{save_path}-results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Fednet Central Accuracy",
            "Fednet Distributed Accuracy",
        ]
    )

for i in range(num_iterations):
    for client in clients:
        client.train(2)

    global_fednet = Net().to(DEVICE)
    client_weights = [client.net.state_dict() for client in clients]
    global_weights = average_weights(client_weights)
    global_fednet.load_state_dict(global_weights)

    fednet_central_accuracy = evaluate(global_fednet, centralised_dataloader)

    fednet_distributed_accuracy = sum(
        [evaluate(client.net, client.testloader) for client in clients]
    ) / len(clients)

    print(f"Fednet Central Accuracy: {fednet_central_accuracy}")
    print(f"Fednet Distributed Accuracy: {fednet_distributed_accuracy}")

    for client in clients:
        client.net.load_state_dict(copy.deepcopy(global_fednet.state_dict()))

    with open(f"{save_path}-results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                fednet_central_accuracy,
                fednet_distributed_accuracy,
            ]
        )
