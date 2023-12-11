import tyro
from enum import Enum, auto
import tqdm
import torch
from torchvision.transforms import ToTensor
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import argparse

import pdb
from data import load_cifar, dataloaders
import wandb

class Split(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


def loop(dataloader, optimizer, model, split, grad_accumulation_steps):
    total_loss = 0
    n = 0
    optimizer.zero_grad()
    for step, images in enumerate(tqdm.tqdm(dataloader)):
        x = images.cuda()
        output = model(x[:,:-1])
        logprobs = output.logits.log_softmax(-1)
        batch_size, length = logprobs.shape
        num_tokens = batch_size * length
        loglik = logprobs[
            torch.arange(batch_size)[:,None,None],
            torch.arange(length)[:,None],
            x[:,1:,None],
        ]
        loss = -loglik.sum()

        if split == Split.TRAIN: 
            (loss / num_tokens / grad_accumulation_steps).backward()
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            wandb.log({"loss": loss})
        
        total_loss += loss.detach()
        n += num_tokens

    if split == Split.TRAIN and grad_accumulation_steps > 1:
        optimizer.step()  # Ensure any remaining gradients are applied
        optimizer.zero_grad()

    return total_loss / n



def main():
    # constants
    parser = argparse.ArgumentParser(description="Train a model on CIFAR with SSM")
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading (default: 4)')
    parser.add_argument('--d-model', type=int, default=256, help='Dimension of model (default: 256)')
    parser.add_argument('--n-layer', type=int, default=4, help='Number of layers (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--grad-accumulation-steps', type=int, default=1, help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--save-model-steps', type=int, default=100, help='Save model every n steps (default: 100)')
    parser.add_argument('--seed', type=int, default=100, help='Save model every n steps (default: 100)')

    args = parser.parse_args()

    # Use args with underscores when accessing the arguments
    batch_size = args.batch_size
    num_workers = args.num_workers
    d_model = args.d_model
    n_layer = args.n_layer
    lr = args.lr
    num_epochs = args.num_epochs
    grad_accumulation_steps = args.grad_accumulation_steps
    save_model_steps = args.save_model_steps
    see = args.seed

    torch.manual_seed(seed)
    vocab_size = 256 + 1

    config = {
        "d_model": d_model,
        "n_layer": n_layer,
        "batchsize": batch_size,
        "lr": lr,
        "num_epochs": num_epochs,
    }
    wandb.init(
        project = "ssm-cifar-tokenized",
        notes = "testing out ssms on tokenized cifar",
        tags = ["ssm", "cifar"],
        config = config,
    )

    data = load_cifar()

    train_loader, valid_loader, test_loader = dataloaders(data, batch_size, num_workers)
    model = MambaLMHeadModel(d_model, n_layer, vocab_size).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # model training
    for epoch in range(num_epochs):
            train_loss = loop(train_loader, optimizer, model, Split.TRAIN, grad_accumulation_steps) # 
            with torch.no_grad():
                valid_loss = loop(valid_loader, optimizer, model, Split.VALID)
            
            wandb.log({
                "train-loss": train_loss,
                "valid-loss": valid_loss,
            })

            if total_steps % save_model_steps == 0:
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'step': total_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }
                torch.save(checkpoint, f'checkpoint_epoch_{epoch}_step_{total_steps}.pth')

    # model test
    with torch.no_grad():
        test_loss = loop(test_loader, optimizer, model, Split.TEST)
    wandb.log({"test-loss": test_loss})



if __name__ == "__main__":
    tyro.cli(main)
