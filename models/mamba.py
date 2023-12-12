import torch
import torch.nn.functional as F
import lightning as L
import wandb
import einops
import torchvision

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import decode
from constants import LOG2, VOCAB_SIZE

class MambaLm(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = MambaLMHeadModel(args.d_model, args.n_layer, VOCAB_SIZE)

        # logging metrics
        self.train_steps = 0 # what if restarting?
        self.epoch = 0

        self.train_loss = 0
        self.train_n = 0
        self.valid_loss = 0
        self.valid_n = 0
        self.test_loss = 0
        self.test_n = 0

    def forward(self, x):
        return self.model(x)

    def nll(self, batch):
        inputs = batch[:, :-1]
        labels = batch[:, 1:]
        output = self.model(inputs)
        nll = F.cross_entropy(
            output.logits.view(-1, VOCAB_SIZE),
            labels.reshape(-1),
            reduction="none",
        )
        return nll

    def bpd(self, batch):
        return self.nll(batch) / LOG2

    # loop functions
    def training_step(self, batch, batch_idx):
        bpd = self.bpd(batch)
        loss = bpd.mean()
        wandb.log(
            {
                "train_step": self.train_steps,
                "train/batch-bpd": loss,
            }
        )
        self.train_steps += 1

        self.train_loss += bpd.sum()
        self.train_n += bpd.shape[0]
        return loss

    def on_train_epoch_end(self):
        wandb.log(
            {
                "train_step": self.train_steps,
                "train/bpd": self.train_loss / self.train_n,
                "epoch": self.epoch,
            }
        )
        self.train_loss = 0
        self.train_n = 0
        self.epoch += 1


    def validation_step(self, batch, batch_idx):
        bpd = self.bpd(batch)
        self.valid_loss += bpd.sum()
        self.valid_n += bpd.shape[0]

    def on_validation_epoch_end(self):
        samples = self.sample_wandb_grid(16)
        wandb.log(
            {
                "train_step": self.train_steps,
                "valid/bpd": self.valid_loss / self.valid_n,
                "samples": samples,
            }
        )
        self.log("valid_bpd", self.valid_loss / self.valid_n)
        self.valid_loss = 0
        self.valid_n = 0

    def test_step(self, batch, batch_idx):
        bpd = self.bpd(batch)
        self.test_loss += bpd.sum()
        self.test_n += bpd.shape[0]

    def on_test_epoch_end(self):
        samples = self.sample_wandb_grid(16)
        wandb.log(
            {
                "train_step": self.train_steps,
                "test/bpd": self.test_loss / self.test_n,
                "samples": samples,
            }
        )
        self.test_loss = 0
        self.test_n = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.args.patience,
            factor=self.args.factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_bpd",
            }
        }

    # sampling helper functions
    @torch.inference_mode()
    def sample(self, num_samples):
        x = torch.full((num_samples, 1), 256, dtype=torch.long, device="cuda:0")
        output = decode(x, self.model, 32 * 32 * 3 + 1, top_k=0)
        return (
            einops.rearrange(
                output.sequences[:, 1:],
                "b (h w c) -> b c h w",
                h=32,
                w=32,
                c=3,
            ).float()
            / 255
        )

    def sample_wandb_grid(self, num_samples):
        samples = self.sample(num_samples)
        image_grid = torchvision.utils.make_grid(samples)
        images = wandb.Image(image_grid)
        return images
