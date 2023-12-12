import os
import math
import tqdm
import torch
import argparse
import wandb
import transformers

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from data import load_cifar, dataloaders
from models.mamba import MambaLm


def main(args):
    L.seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "".join(f"{k[:2]}{v}" for k, v in vars(args).items())
    checkpoint_path = f"checkpoints/{model_name}"
    os.makedirs(checkpoint_path, exist_ok=True)

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
    model = MambaLm(args).to(device)

    trainer = L.Trainer(
        default_root_dir=checkpoint_path,
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=args.num_epochs,
        max_steps=args.train_steps,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="valid_bpd"),
            LearningRateMonitor("epoch")
        ],
    )

    trainer.fit(model, train_loader, valid_loader)
    valid_result = trainer.validate(model, valid_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)

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
        "--patience", type=int, default=4, help="Number of plateau epochs (default: 4)"
    )
    parser.add_argument(
        "--factor", type=float, default=0.5, help="Decay factor after plateau (default: 0.5)"
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
