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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NKDLoss(nn.Module):
    """PyTorch version of NKD"""

    def __init__(
        self,
        temp=1.0,
        gamma=1.5,
    ):
        super(NKDLoss, self).__init__()

        self.temp = temp
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logit_s, logit_t, gt_label):

        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        N, c = logit_s.shape
        s_i = self.log_softmax(logit_s)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        loss_t = -(t_t * s_t).mean()

        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
        logit_s = logit_s[mask].reshape(N, -1)
        logit_t = logit_t[mask].reshape(N, -1)

        # N*class
        S_i = self.log_softmax(logit_s / self.temp)
        T_i = F.softmax(logit_t / self.temp, dim=1)

        loss_non = (T_i * S_i).sum(dim=1).mean()
        loss_non = -self.gamma * (self.temp**2) * loss_non

        return loss_t + loss_non


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


num_clients = 10
num_iterations = 100

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
        self.fednet = Net().to(DEVICE)
        self.distillnet = Net().to(DEVICE)
        self.fedoptimizer = optim.SGD(self.fednet.parameters(), lr=0.01, momentum=0.9)
        self.distilloptimizer = optim.SGD(
            self.distillnet.parameters(), lr=0.01, momentum=0.9
        )
        self.fedcriterion = nn.CrossEntropyLoss()
        self.distillcriterion = NKDLoss()
        self.trainloader = trainloader
        self.testloader = testloader

    def train(self, epochs):
        for _ in tqdm(range(epochs), desc="Train"):
            self.fednet.train()
            for batch in tqdm(self.trainloader, leave=False):
                inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                self.fedoptimizer.zero_grad()
                outputs = self.fednet(inputs)
                loss = self.fedcriterion(outputs, labels)
                loss.backward()
                self.fedoptimizer.step()

        # print(f"DEBUG: {evaluate(self.fednet, self.testloader)}")

    def distill(self, epochs):
        for _ in tqdm(range(epochs), desc="Distill"):
            self.distillnet.train()
            for batch in tqdm(self.trainloader, leave=False):
                inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                self.distilloptimizer.zero_grad()
                outputs = self.distillnet(inputs)
                loss = self.distillcriterion(outputs, self.fednet(inputs), labels)
                loss.backward()
                self.distilloptimizer.step()

        # print(f"DEBUG: {evaluate(self.fednet, self.testloader)}")


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

save_path = "outputs/imagenet-niid-fedmix-nkd-vgg-" + datetime.now().strftime(
    "%d-%m-%H-%M"
)

with open(f"{save_path}-fedmix-local-results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Fednet Central Accuracy",
            "Fednet Distributed Accuracy",
            "Distillnet Central Accuracy",
            "Distillnet Distributed Accuracy",
        ]
    )

for i in range(num_iterations):
    for client in clients:
        client.train(2)

    global_fednet = Net().to(DEVICE)
    client_weights = [client.fednet.state_dict() for client in clients]
    global_weights = average_weights(client_weights)
    global_fednet.load_state_dict(global_weights)

    fednet_central_accuracy = evaluate(global_fednet, centralised_dataloader)

    fednet_distributed_accuracy = sum(
        [evaluate(client.fednet, client.testloader) for client in clients]
    ) / len(clients)

    print(f"Fednet Central Accuracy: {fednet_central_accuracy}")
    print(f"Fednet Distributed Accuracy: {fednet_distributed_accuracy}")

    for client in clients:
        client.fednet.load_state_dict(copy.deepcopy(global_fednet.state_dict()))

    # if i > 20:
    for client in clients:
        client.distill(2)

    distillnet_distributed_accuracy = sum(
        [evaluate(client.distillnet, client.testloader) for client in clients]
    ) / len(clients)

    distillnet_central_accuracy = sum(
        [evaluate(client.distillnet, centralised_dataloader) for client in clients]
    ) / len(clients)
    # else:
    #     distillnet_distributed_accuracy = 0
    #     distillnet_central_accuracy = 0

    print(f"Distillnet Central Accuracy: {distillnet_central_accuracy}")
    print(f"Distillnet Distributed Accuracy: {distillnet_distributed_accuracy}")

    with open(f"{save_path}-fedmix-local-results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                fednet_central_accuracy,
                fednet_distributed_accuracy,
                distillnet_central_accuracy,
                distillnet_distributed_accuracy,
            ]
        )
