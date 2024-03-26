from collections import OrderedDict

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

import warnings
from typing import Dict, List, Optional, Union

import numpy as np

import datasets
from flwr.common.typing import NDArrayFloat
from flwr_datasets.partitioner.partitioner import Partitioner


# pylint: disable=R0902, R0912
class DirichletPartitioner(Partitioner):
    """Partitioner based on Dirichlet distribution.
    Implementation based on Bayesian Nonparametric Federated Learning of Neural Networks
    https://arxiv.org/abs/1905.12022.
    The algorithm sequentially divides the data with each label. The fractions of the
    data with each label is drawn from Dirichlet distribution and adjusted in case of
    balancing. The data is assigned. In case the `min_partition_size` is not satisfied
    the algorithm is run again (the fractions will change since it is a random process
    even though the alpha stays the same).
    The notion of balancing is explicitly introduced here (not mentioned in paper but
    implemented in the code). It is a mechanism that excludes the node from
    assigning new samples to it if the current number of samples on that node exceeds
    the average number that the node would get in case of even data distribution.
    It is controlled by`self_balancing` parameter.
    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    partition_by : str
        Column name of the labels (targets) based on which Dirichlet sampling works.
    alpha : Union[int, float, List[float], NDArrayFloat]
        Concentration parameter to the Dirichlet distribution
    min_partition_size : int
        The minimum number of samples that each partitions will have (the sampling
        process is repeated if any partition is too small).
    self_balancing : bool
        Whether assign further samples to a partition after the number of samples
        exceeded the average number of samples per partition. (True in the original
        paper's code although not mentioned in paper itself).
    shuffle: bool
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to nodes.
    seed: int
        Seed used for dataset shuffling. It has no effect if `shuffle` is False.
    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>>
    >>> partitioner = DirichletPartitioner(num_partitions=10, partition_by="label",
    >>>                                    alpha=0.5, min_partition_size=10,
    >>>                                    self_balancing=True)
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    >>> print(partition[0])  # Print the first example
    {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x127B92170>,
    'label': 4}
    >>> partition_sizes = [len(fds.load_partition(node_id)) for node_id in range(10)]
    >>> print(sorted(partition_sizes))
    [2134, 2615, 3646, 6011, 6170, 6386, 6715, 7653, 8435, 10235]
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_partitions: int,
        partition_by: str,
        alpha: Union[int, float, List[float], NDArrayFloat],
        min_partition_size: int = 10,
        self_balancing: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._alpha: NDArrayFloat = self._initialize_alpha(alpha)
        self._partition_by = partition_by
        self._min_partition_size: int = min_partition_size
        self._self_balancing = self_balancing
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._avg_num_of_samples_per_node: Optional[float] = None
        self._unique_classes: Optional[Union[List[int], List[str]]] = None
        self._node_id_to_indices: Dict[int, List[int]] = {}
        self._node_id_to_indices_determined = False

    def load_partition(self, node_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.
        Parameters
        ----------
        node_id : int
            the index that corresponds to the requested partition
        Returns
        -------
        dataset_partition : Dataset
            single partition of a dataset
        """
        # The partitioning is done lazily - only when the first partition is
        # requested. Only the first call creates the indices assignments for all the
        # partition indices.
        self._check_num_partitions_correctness_if_needed()
        self._determine_node_id_to_indices_if_needed()
        return self.dataset.select(self._node_id_to_indices[node_id])

    def _initialize_alpha(
        self, alpha: Union[int, float, List[float], NDArrayFloat]
    ) -> NDArrayFloat:
        """Convert alpha to the used format in the code a NDArrayFloat.
        The alpha can be provided in constructor can be in different format for user
        convenience. The format into which it's transformed here is used throughout the
        code for computation.
        Parameters
        ----------
            alpha : Union[int, float, List[float], NDArrayFloat]
                Concentration parameter to the Dirichlet distribution
        Returns
        -------
        alpha : NDArrayFloat
            Concentration parameter in a format ready to used in computation.
        """
        if isinstance(alpha, int):
            alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, float):
            alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, List):
            if len(alpha) != self._num_partitions:
                raise ValueError(
                    "If passing alpha as a List, it needs to be of length of equal to "
                    "num_partitions."
                )
            alpha = np.asarray(alpha)
        elif isinstance(alpha, np.ndarray):
            # pylint: disable=R1720
            if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
                raise ValueError(
                    "If passing alpha as an NDArray, its length needs to be of length "
                    "equal to num_partitions."
                )
            elif alpha.ndim == 2:
                alpha = alpha.flatten()
                if alpha.shape[0] != self._num_partitions:
                    raise ValueError(
                        "If passing alpha as an NDArray, its size needs to be of length"
                        " equal to num_partitions."
                    )
        else:
            raise ValueError("The given alpha format is not supported.")
        if not (alpha > 0).all():
            raise ValueError(
                f"Alpha values should be strictly greater than zero. "
                f"Instead it'd be converted to {alpha}"
            )
        return alpha

    def _determine_node_id_to_indices_if_needed(self) -> None:  # pylint: disable=R0914
        """Create an assignment of indices to the partition indices."""
        if self._node_id_to_indices_determined:
            return

        # Generate information needed for Dirichlet partitioning
        self._unique_classes = self.dataset.unique(self._partition_by)
        assert self._unique_classes is not None
        # This is needed only if self._self_balancing is True (the default option)
        self._avg_num_of_samples_per_node = self.dataset.num_rows / self._num_partitions

        # Change targets list data type to numpy
        targets = np.array(self.dataset[self._partition_by])

        # Repeat the sampling procedure based on the Dirichlet distribution until the
        # min_partition_size is reached.
        sampling_try = 0
        while True:
            # Prepare data structure to store indices assigned to node ids
            node_id_to_indices: Dict[int, List[int]] = {}
            for nid in range(self._num_partitions):
                node_id_to_indices[nid] = []

            # Iterated over all unique labels (they are not necessarily of type int)
            for k in self._unique_classes:
                # Access all the indices associated with class k
                indices_representing_class_k = np.nonzero(targets == k)[0]
                # Determine division (the fractions) of the data representing class k
                # among the partitions
                class_k_division_proportions = self._rng.dirichlet(self._alpha)
                nid_to_proportion_of_k_samples = {}
                for nid in range(self._num_partitions):
                    nid_to_proportion_of_k_samples[nid] = class_k_division_proportions[
                        nid
                    ]
                # Balancing (not mentioned in the paper but implemented)
                # Do not assign additional samples to the node if it already has more
                # than the average numbers of samples per partition. Note that it might
                # especially affect classes that are later in the order. This is the
                # reason for more sparse division that the alpha might suggest.
                if self._self_balancing:
                    assert self._avg_num_of_samples_per_node is not None
                    for nid in nid_to_proportion_of_k_samples.copy():
                        if (
                            len(node_id_to_indices[nid])
                            > self._avg_num_of_samples_per_node
                        ):
                            nid_to_proportion_of_k_samples[nid] = 0

                    # Normalize the proportions such that they sum up to 1
                    sum_proportions = sum(nid_to_proportion_of_k_samples.values())
                    for nid, prop in nid_to_proportion_of_k_samples.copy().items():
                        nid_to_proportion_of_k_samples[nid] = prop / sum_proportions

                # Determine the split indices
                cumsum_division_fractions = np.cumsum(
                    list(nid_to_proportion_of_k_samples.values())
                )
                cumsum_division_numbers = cumsum_division_fractions * len(
                    indices_representing_class_k
                )
                # [:-1] is because the np.split requires the division indices but the
                # last element represents the sum = total number of samples
                indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]

                split_indices = np.split(
                    indices_representing_class_k, indices_on_which_split
                )

                # Append new indices (coming from class k) to the existing indices
                for nid, indices in node_id_to_indices.items():
                    indices.extend(split_indices[nid].tolist())

            # Determine if the indices assignment meets the min_partition_size
            # If it does not mean the requirement repeat the Dirichlet sampling process
            # Otherwise break the while loop
            min_sample_size_on_client = min(
                len(indices) for indices in node_id_to_indices.values()
            )
            if min_sample_size_on_client >= self._min_partition_size:
                break
            sample_sizes = [len(indices) for indices in node_id_to_indices.values()]
            alpha_not_met = [
                self._alpha[i]
                for i, ss in enumerate(sample_sizes)
                if ss == min(sample_sizes)
            ]
            mssg_list_alphas = (
                (
                    "Generating partitions by sampling from a list of very wide range "
                    "of alpha values can be hard to achieve. Try reducing the range "
                    f"between maximum ({max(self._alpha)}) and minimum alpha "
                    f"({min(self._alpha)}) values or increasing all the values."
                )
                if len(self._alpha.flatten().tolist()) > 0
                else ""
            )
            warnings.warn(
                f"The specified min_partition_size ({self._min_partition_size}) was "
                f"not satisfied for alpha ({alpha_not_met}) after "
                f"{sampling_try} attempts at sampling from the Dirichlet "
                f"distribution. The probability sampling from the Dirichlet "
                f"distribution will be repeated. Note: This is not a desired "
                f"behavior. It is recommended to adjust the alpha or "
                f"min_partition_size instead. {mssg_list_alphas}",
                stacklevel=1,
            )
            if sampling_try == 10:
                raise ValueError(
                    "The max number of attempts (10) was reached. "
                    "Please update the values of alpha and try again."
                )
            sampling_try += 1

        # Shuffle the indices not to have the datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in node_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)
        self._node_id_to_indices = node_id_to_indices
        self._node_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._node_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _check_num_partitions_greater_than_zero(self) -> None:
        """Test num_partition left sides correctness."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")


def apply_transforms(batch):
    """Get transformation for MNIST dataset."""
    tfs = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    batch["image"] = [tfs(img) for img in batch["image"]]
    return batch


class Net(nn.Module):
    def __init__(self, num_classes=101):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(7 * 7 * 512, 4096), nn.ReLU()
        )
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


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
            images, labels = batch["image"].to(device), batch["label"].to(device)
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

    # _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    # axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    # axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

    # axs[0].set_ylabel("Loss")
    # axs[1].set_ylabel("Accuracy")

    # # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    # plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.plot(np.asarray(rounds_loss), np.asarray(values))

    plt.ylabel("Accuracy")

    # plt.ylim(0, 1)
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
    NUM_CLIENTS = 3

    mnist_fds = FederatedDataset(
        dataset="food101",
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=3, alpha=0.5, partition_by="label"
            ),
        },
    )
    centralized_testset = mnist_fds.load_full("validation")

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
        config=fl.server.ServerConfig(num_rounds=1),
        client_resources={"num_cpus": 1, "num_gpus": 0},
        strategy=strategy,
        actor_kwargs={"on_actor_init_fn": disable_progress_bar},
    )

    print("................")
    print(history)

    save_path = (
        "outputs/" + datetime.now().strftime("%d-%m-%H-%M") + "-food101-niid-fedntd"
    )

    save_results_as_pickle(history, file_path=save_path, extra_results={})
    plot_metric_from_history(
        history,
        save_path,
    )


if __name__ == "__main__":
    main()
