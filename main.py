import os
import math
import tqdm
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import argparse
import wandb
import transformers

from data import load_cifar, dataloaders
import sample


def evaluate(dataloader, model, num_samples):
    total_loss = 0
    n = 0

    for step, images in enumerate(tqdm.tqdm(dataloader)):
        x = images.cuda()
        output = model(x[:, :-1])
        logprobs = output.logits.log_softmax(-1)
        batch_size, length, vocab = logprobs.shape
        loglik = logprobs.gather(-1, x[:, 1:, None])
        loss = -loglik.sum()

        # loss accounting
        num_tokens = batch_size * length
        n += num_tokens
        total_loss += loss.detach()

    # TODO evaluate generated images
    samples = sample.sample_wandb_grid(model, num_samples)
    return total_loss / n, samples


def train(
    dataloader,
    optimizer,
    scheduler,
    model,
    start_step,
    grad_accumulation_steps=1,
    train_steps=-1,
):
    total_loss = 0
    total_step = 0
    n = 0
    batch_loss = 0
    batch_n = 0
    optimizer.zero_grad()
    for step, images in enumerate(tqdm.tqdm(dataloader)):
        if train_steps > 0 and step > train_steps:
            break
        x = images.cuda()
        output = model(x[:, :-1])
        logprobs = output.logits.log_softmax(-1)
        batch_size, length, vocab = logprobs.shape
        loglik = logprobs.gather(-1, x[:, 1:, None])
        loss = -loglik.sum()

        # loss accounting
        num_tokens = batch_size * length
        batch_n += num_tokens
        batch_loss += loss.detach()
        n += num_tokens
        total_loss += loss.detach()

        # update weights
        # average over total number of pixels*channels in batch
        (loss / num_tokens).backward()
        if (step + 1) % grad_accumulation_steps == 0:
            wandb.log(
                {
                    "train_step": start_step + total_step,
                    "train/batch-loss": batch_loss / batch_n,
                    "train/batch-bpd": batch_loss / batch_n / math.log(2),
                }
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            batch_n = 0
            batch_loss = 0
            total_step += 1

    if (grad_accumulation_steps > 1) and (step % grad_accumulation_steps != 0):
        optimizer.step()  # Ensure any remaining gradients are applied
        scheduler.step()
        optimizer.zero_grad()
        total_step += 1

    return total_loss / n, total_step


def main(args):
    # constants
    torch.manual_seed(args.seed)
    vocab_size = 256 + 1

    wandb.init(
        project="ssm-cifar-tokenized",
        notes="testing out ssms on tokenized cifar",
        tags=["ssm", "cifar"],
        config=args,
    )

    # wandb can log only once per step by default, define custom step
    wandb.define_metric("train_step")
    wandb.define_metric("*", step_metric="train_step")

    data = load_cifar()

    train_loader, valid_loader, test_loader = dataloaders(
        data, args.batch_size, args.num_workers
    )
    model = MambaLMHeadModel(args.d_model, args.n_layer, vocab_size).cuda()

    # optimizer hyperparams from
    # https://github.com/state-spaces/s4/blob/main/configs/experiment/cifar/s4-cifar.yaml
    # https://github.com/state-spaces/s4/blob/main/configs/optimizer/adamw.yaml
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        # len(training_loader)
        num_warmup_steps=len(train_loader),
        # len(training_loader) * args.num_epochs
        num_training_steps=len(train_loader)*args.num_epochs,
    )

    total_step = 0
    best_valid_loss = 1e10

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        # train
        model.train()
        train_loss, train_steps = train(
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            start_step=total_step,
            grad_accumulation_steps=args.grad_accumulation_steps,
            train_steps=args.train_steps,
        )
        total_step += train_steps
        wandb.log(
            {
                "epoch": epoch,
                "train_step": total_step,
                "train/loss": train_loss,
                "train/bpd": train_loss / math.log(2),
            }
        )

        # validate
        model.eval()
        with torch.no_grad():
            valid_loss, valid_samples = evaluate(
                valid_loader,
                model,
                num_samples=args.num_samples,
            )
        wandb.log(
            {
                "train_step": total_step,
                "val/loss": valid_loss,
                "val/bpd": valid_loss / math.log(2),
                "samples": valid_samples,
            }
        )

        if valid_loss < best_valid_loss:
            print(f"New best valid loss: {valid_loss}, saving model")
            best_valid_loss = valid_loss
            # Save checkpoint
            checkpoint = {
                "args": args,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }
            model_name = "".join(f"{k[:2]}{v}" for k, v in vars(args).items())
            os.makedirs(f"checkpoints/{model_name}", exist_ok=True)
            torch.save(
                checkpoint, f"checkpoints/{model_name}/checkpoint_epoch_{epoch}.pth"
            )

    # test
    model.eval()
    with torch.no_grad():
        test_loss, test_samples = evaluate(
            test_loader, model, num_samples=args.num_samples
        )
    wandb.log(
        {
            "train_step": total_step,
            "test/loss": test_loss,
            "test/bpd": test_loss / math.log2(math.exp(1)),
            "samples": test_samples,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR with SSM")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for training (default: 50)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--d-model", type=int, default=512, help="Dimension of model (default: 512)"
    )
    parser.add_argument(
        "--n-layer", type=int, default=6, help="Number of layers (default: 6)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Learning rate (default: 1e-2)"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=200, help="Number of epochs (default: 200)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples at eval (default: 16)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=-1,
        help="Number of batches to run in training. Debugging only. (default: -1)",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--save-model-steps",
        type=int,
        default=500,
        help="Save model every n steps (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Save model every n steps (default: 1234)",
    )

    args = parser.parse_args()

    main(args)
