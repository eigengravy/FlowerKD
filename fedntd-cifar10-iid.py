from collections import OrderedDict

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader


import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torchvision.transforms import Compose, Normalize, ToTensor

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


def apply_transforms(batch):
    """Get transformation for MNIST dataset."""
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


class Net(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, xb):
        return self.network(xb)


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    tau: float,
    beta: float,
    num_classes: int,
) -> None:
    """Train the network on the training set.
    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    tau : float
        Parameter for tau.
    beta : float
        Parameter for beta.
    """
    criterion = NTDLoss(num_classes=num_classes, tau=tau, beta=beta)
    global_net = Net(num_classes).to(device=device)
    global_net.load_state_dict(net.state_dict())
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            local_logits = net(images)
            with torch.no_grad():
                global_logits = global_net(images)
            loss = criterion(local_logits, labels, global_logits)
            loss.backward()
            optimizer.step()


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.
    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.
    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["img"].to(device), data["label"].to(device)
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


class NTDLoss(nn.Module):
    """Not-true Distillation Loss.
    As described in:
    [Preservation of the Global Knowledge by Not-True Distillation in Federated Learning](https://arxiv.org/pdf/2106.03097.pdf)
    """

    def __init__(self, num_classes=10, tau=3, beta=1):
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


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(self, trainloader, valloader) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes=10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def set_parameters(self, parameters):
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        lr, epochs = config["lr"], config["epochs"]
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        train(
            self.model,
            self.trainloader,
            optim,
            epochs=epochs,
            device=self.device,
            tau=1.0,
            beta=1.0,
            num_classes=10,
        )
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}


def get_client_fn(dataset: FederatedDataset):
    """Return a function to construct a client.
    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
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
    """Generate the function for centralized evaluation.
    Parameters
    ----------
    centralized_testset : Dataset
        The dataset to test the model with.
    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate_fn(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = Net(num_classes=10)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        testset = centralized_testset.with_transform(apply_transforms)
        testloader = DataLoader(testset, batch_size=50)
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

    _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()


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


def main() -> None:
    NUM_CLIENTS = 10

    mnist_fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    centralized_testset = mnist_fds.load_full("test")

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config: Dict[str, Scalar] = {
            "epochs": 1,
            "lr": 0.01,
        }
        return config

    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(centralized_testset),
    )

    disable_progress_bar()

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(mnist_fds),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
        actor_kwargs={"on_actor_init_fn": disable_progress_bar},
    )

    print("................")
    print(history)

    save_path = datetime.now().strftime("%d-%m-%H-%M") + "-fedntd-cifar10-iid"

    save_results_as_pickle(history, file_path=save_path, extra_results={})
    plot_metric_from_history(
        history,
        save_path,
    )


if __name__ == "__main__":
    main()
