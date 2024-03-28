from collections import OrderedDict
import copy
import csv
import os

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader


import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torchvision import transforms
from typing import Callable, Dict, Optional, Tuple, Union

from datasets import Dataset

import pickle
from pathlib import Path
from secrets import token_hex

import matplotlib.pyplot as plt
import numpy as np
from flwr.server.history import History


from datasets.utils.logging import disable_progress_bar
from datetime import datetime


from flwr_datasets.partitioner import DirichletPartitioner


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


class NTDLoss(nn.Module):
    def __init__(self, num_classes=200, tau=3, beta=1):
        super(NTDLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, local_logits, targets, global_logits):
        """Forward pass."""
        ce_loss = self.CE(local_logits, targets)
        local_logits = self._refine_as_not_true(local_logits, targets)
        local_probs = F.log_softmax(local_logits / self.tau, dim=1)
        with torch.no_grad():
            global_logits = self._refine_as_not_true(global_logits, targets)
            global_probs = torch.softmax(global_logits / self.tau, dim=1)

        ntd_loss = (self.tau**2) * self.KLDiv(local_probs, global_probs)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _refine_as_not_true(
        self,
        logits,
        targets,
    ):
        nt_positions = torch.arange(0, self.num_classes).to(logits.device)
        nt_positions = nt_positions.repeat(logits.size(0), 1)
        nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
        nt_positions = nt_positions.view(-1, self.num_classes - 1)

        logits = torch.gather(logits, 1, nt_positions)

        return logits


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
):
    criterion = nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def distill(
    student: nn.Module,
    teacher: nn.Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    tau: float,
    beta: float,
    num_classes: int,
) -> None:
    criterion = NTDLoss(num_classes=num_classes, tau=tau, beta=beta)
    student.train()
    teacher.eval()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            local_logits = student(images)
            with torch.no_grad():
                global_logits = teacher(images)
            loss = criterion(local_logits, labels, global_logits)
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


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(self, trainloader, valloader) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fednet = Net(num_classes=200).to(self.device)
        self.distillnet = Net(num_classes=200).to(self.device)

    def set_parameters(self, net, parameters):
        """Change the parameters of the model using the given ones."""
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.fednet.state_dict().items()]

    def fit(self, parameters, config):
        # self.set_parameters(self.fednet, parameters) ?? TODO: neeeded? confirm
        teacher = Net(num_classes=200).to(self.device)
        self.set_parameters(teacher, parameters)
        lr, epochs = config["lr"], config["epochs"]
        optim = torch.optim.SGD(self.fednet.parameters(), lr=lr, momentum=0.9)
        train(
            self.fednet,
            self.trainloader,
            optim,
            epochs=epochs,
            device=self.device,
        )
        optim = torch.optim.SGD(self.distillnet.parameters(), lr=lr, momentum=0.9)
        distill(
            self.distillnet,
            teacher,
            self.trainloader,
            optim,
            epochs=epochs,
            tau=3,
            beta=1,
            num_classes=200,
            device=self.device,
        )
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        loss, accuracy = test(self.distillnet, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}


def get_client_fn(dataset: FederatedDataset):
    def client_fn(cid: str) -> fl.client.Client:
        client_dataset = dataset.load_partition(int(cid), "train")
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1)
        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]
        trainloader = DataLoader(
            trainset.with_transform(apply_transforms), batch_size=32, shuffle=True
        )
        valloader = DataLoader(valset.with_transform(apply_transforms), batch_size=32)
        return FlowerClient(trainloader, valloader).to_client()

    return client_fn


def get_evaluate_fn(
    centralized_testset: Dataset,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:

    def evaluate_fn(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = Net(num_classes=200)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict)
        testset = centralized_testset.with_transform(apply_transforms)
        testloader = DataLoader(testset, batch_size=32)
        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn


def plot_metric_from_history(
    hist: History,
    save_plot_path: str,
    suffix: Optional[str] = "",
) -> None:
    """Plot from Flower server History.
    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : str
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    _, values = zip(*metric_dict["accuracy"])

    # let's extract centralised loss (main metric reported in FedProx paper)
    rounds_loss, values_loss = zip(*hist.losses_centralized)

    # _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    # axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    # axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    # axs[0].set_ylabel("Loss")
    # axs[1].set_ylabel("Accuracy")

    # axs[1].set_ylim(0, 1)

    plt.plot(np.asarray(rounds_loss), np.asarray(values))

    plt.ylabel("Accuracy")

    plt.ylim(0, 1)
    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()

    print(hist.losses_distributed)
    print(hist.metrics_distributed["accuracy"])
    curr_time = datetime.now().strftime("%d-%m-%H-%M")
    os.mkdir(f"outputs/imagenet-niid-fedmix-{curr_time}")
    with open(
        f"outputs/imagenet-niid-fedmix-{curr_time}/results.csv",
        "w",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["loss", "accuracy"])
        (_, v_loss), (_, v_accuracy) = zip(
            *hist.losses_distributed, *hist.metrics_distributed["accuracy"]
        )
        writer.writerows(zip(v_loss, v_accuracy))


def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_results: Optional[Dict] = None,
    default_filename: str = "results.pkl",
) -> None:
    """Save results from simulation to pickle.
    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwritting results.
    extra_results : Optional[Dict]
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """
    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Add a randomly generated suffix to the file name (so it doesn't.
        overwrite the file).
        """
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Append the default file name to the path."""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        # file exists already
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")

    data = {"history": history}
    if extra_results is not None:
        data = {**data, **extra_results}

    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self, server_round: int, results, failures
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(
            f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}"
        )

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}


def main() -> None:
    NUM_CLIENTS = 10

    mnist_fds = FederatedDataset(
        dataset="zh-plus/tiny-imagenet",
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=NUM_CLIENTS, alpha=0.5, partition_by="label"
            ),
        },
    )
    centralized_testset = mnist_fds.load_split("valid")

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config: Dict[str, Scalar] = {
            "epochs": 1,
            "lr": 0.01,
        }
        return config

    strategy = AggregateCustomMetricStrategy(
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(centralized_testset),
    )

    disable_progress_bar()

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(mnist_fds),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=1),
        client_resources={"num_cpus": 4, "num_gpus": 2},
        strategy=strategy,
        actor_kwargs={"on_actor_init_fn": disable_progress_bar},
    )

    print("................")
    print(history)

    save_path = "outputs/imagenet-niid-fedntd-" + datetime.now().strftime("%d-%m-%H-%M")

    save_results_as_pickle(history, file_path=save_path, extra_results={})
    plot_metric_from_history(
        history,
        save_path,
    )


if __name__ == "__main__":
    main()
