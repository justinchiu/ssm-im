import tyro
from enum import Enum, auto
import tqdm
import torch
from torchvision.transforms import ToTensor
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import pdb
from data import load_cifar, dataloaders
import wandb

class Split(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def loop(dataloader, optimizer, model, split):
    total_loss = 0
    n = 0
    for images in tqdm.tqdm(dataloader):
        if split == Split.TRAIN:
            optimizer.zero_grad()
        x = images.cuda()
        batch_size, length = x.shape

        output = model(x[:,:-1])
        loss = -output.logits.log_softmax(-1)[
            torch.arange(batch_size)[:,None,None],
            torch.arange(length-1)[:,None],
            x[:,1:,None],
        ].mean()
        if split == Split.TRAIN: 
            loss.backward()
            optimizer.step()
            wandb.log({
                "loss": loss,
            })
        total_loss += loss
        n += 1
    return total_loss / n


def main():
    torch.manual_seed(1234)
    batch_size = 128
    num_workers = 4
    greyscale = False
    d_model = 256
    n_layer = 4
    vocab_size = 256 + 1
    lr = 1e-3
    num_epochs = 5

    wandb.init(
        project = "ssm-cifar-tokenized",
        notes = "testing out ssms on tokenized cifar",
        tags = ["ssm", "cifar"],
    )
    wandb.config = {
        "d_model": d_model,
        "n_layer": n_layer,
        "batchsize": batch_size,
        "lr": lr,
        "num_epochs": num_epochs,
    }

    data = load_cifar(greyscale)

    train_loader, valid_loader, test_loader = dataloaders(data, batch_size, num_workers)
    model = MambaLMHeadModel(d_model, n_layer, vocab_size).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = loop(train_loader, optimizer, model, Split.TRAIN)
        valid_loss = loop(valid_loader, optimizer, model, Split.VALID)
        wandb.log({
            "train-loss": train_loss,
            "valid-loss": valid_loss,
        })
    test_loss = loop(test_loader, optimizer, model, Split.TEST)
    wandb.log({"test-loss": test_loss})



if __name__ == "__main__":
    tyro.cli(main)
