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


def loop(dataloader, optimizer, model, split, grad_accumulation_steps=1):
    total_loss = 0
    n = 0
    batch_n = 0
    optimizer.zero_grad()
    for step, images in enumerate(tqdm.tqdm(dataloader)):
        x = images.cuda()
        output = model(x[:,:-1])
        logprobs = output.logits.log_softmax(-1)
        batch_size, length = logprobs.shape
        loglik = logprobs[
            torch.arange(batch_size)[:,None,None],
            torch.arange(length)[:,None],
            x[:,1:,None],
        ]
        loss = -loglik.sum()

        # loss accounting
        num_tokens = batch_size * length
        batch_n += num_tokens
        n += num_tokens
        total_loss += loss.detach()

        # update weights
        if split == Split.TRAIN:
            # average over total number of pixels*channels in batch
            (loss / batch_n).backward()
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_n = 0

            wandb.log({
                "train NLL loss": loss,
                "train bpd": loss / math.log2(math.exp(1))
            })
        

    if split == Split.TRAIN and grad_accumulation_steps > 1:
        optimizer.step()  # Ensure any remaining gradients are applied
        optimizer.zero_grad()

    return total_loss / n



def main():
    # constants
    torch.manual_seed(args.seed)
    vocab_size = 256 + 1

    wandb.init(
        project = "ssm-cifar-tokenized",
        notes = "testing out ssms on tokenized cifar",
        tags = ["ssm", "cifar"],
        config = args,
    )

    data = load_cifar()

    train_loader, valid_loader, test_loader = dataloaders(data, args.batch_size, args.num_workers)
    model = MambaLMHeadModel(args.d_model, args.n_layer, vocab_size).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # model training
    for epoch in range(args.num_epochs):
        train_loss = loop(
            train_loader,
            optimizer,
            model,
            Split.TRAIN,
            grad_accumulation_steps=args.grad_accumulation_steps,
        )
        with torch.no_grad():
            valid_loss = loop(
                valid_loader,
                optimizer,
                model,
                Split.VALID
            )
        
        wandb.log({
            "train-loss": train_loss,
            "valid-loss": valid_loss,
            "train-bpd": train_loss / math.log2(math.exp(1)),
            "valid-bpd": valid_loss / math.log2(math.exp(1)),
        })

        if total_steps % args.save_model_steps == 0:
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
    wandb.log({
        "test-loss": test_loss,
        "test-bpd": test_loss / math.log2(math.exp(1)),
    })



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR with SSM")
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading (default: 4)')
    parser.add_argument('--d-model', type=int, default=256, help='Dimension of model (default: 256)')
    parser.add_argument('--n-layer', type=int, default=8, help='Number of layers (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs (default: 5)')
    parser.add_argument('--grad-accumulation-steps', type=int, default=1, help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--save-model-steps', type=int, default=100, help='Save model every n steps (default: 100)')
    parser.add_argument('--seed', type=int, default=1234, help='Save model every n steps (default: 100)')

    args = parser.parse_args()

    main(args)
