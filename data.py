from dataclasses import dataclass
import einops
import einops.layers.torch
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


@dataclass
class Data:
    train: torch.utils.data.Dataset
    valid: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
    d_input: int
    d_output: int


def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0 - val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


def load_cifar(grayscale=False):
    if grayscale:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
                torchvision.transforms.Lambda(lambda x: x.view(1, 1024).t()),
            ]
        )
    else:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
                torchvision.transforms.Lambda(lambda x: x.view(3, 1024).t()),
            ]
        )
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: (x * 255).long()),
            einops.layers.torch.Rearrange("... c h w -> ... (h w c)"),
        ]
    )

    # S4 is trained on sequences with no data augmentation!
    transform_train = transform_test = transform

    trainset = torchvision.datasets.CIFAR10(
        root="./data/cifar/", train=True, download=True, transform=transform_train
    )
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.CIFAR10(
        root="./data/cifar/", train=True, download=True, transform=transform_test
    )
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.CIFAR10(
        root="./data/cifar/", train=False, download=True, transform=transform_test
    )

    d_input = 3 if not grayscale else 1
    d_output = 10

    return Data(
        train=trainset,
        valid=valset,
        test=testset,
        d_input=d_input,
        d_output=d_output,
    )

def dataloaders(data, batch_size, num_workers, device="cuda:0"):
    def collate_fn(batch):
        stacked_images = torch.stack(
            [img for img, label in batch]
        )
        batch, length = stacked_images.shape
        return torch.concatenate([
            torch.full((batch, 1), 256),
            stacked_images,
        ], dim=1)

        return torch.utils.data.default_collate(
        )
    
    # Dataloaders
    trainloader = torch.utils.data.DataLoader(
        data.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device=device,
        collate_fn=collate_fn,
    )
    valloader = torch.utils.data.DataLoader(
        data.valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device=device,
        collate_fn=collate_fn,
    )
    testloader = torch.utils.data.DataLoader(
        data.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device=device,
        collate_fn=collate_fn,
    )
    return trainloader, valloader, testloader

