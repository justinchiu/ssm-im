import time
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import decode
import einops
import torchvision
import wandb



if __name__ == "__main__":
    start = time.time()

    # load model
    checkpoint_dict = torch.load(
        "checkpoints/ba64nu4d_256n_8lr0.001nu50gr2sa500se1234/checkpoint_epoch_2.pth"
    )
    model = MambaLMHeadModel(256, 8, 257)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    model.cuda()

    samples = sample_wandb_grid(model, 16)

    torch.cuda.synchronize("cuda:0")
    print("Took", time.time() - start, "secs")

    wandb.init(
        project="ssm-image-generation-test",
        notes="testing mamba image generation",
        tags=["ssm", "cifar"],
    )

    wandb.log({"samples": samples})
