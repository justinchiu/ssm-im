import time
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import decode
import einops
import torchvision
import wandb

@torch.inference_mode()
def naive_sample(model, num_samples):
    x = torch.full((num_samples, 1), 256, dtype=torch.long, device="cuda:0")
    output = decode(x, model, 32*32*3+1, top_k=0)
    return einops.rearrange(
        output.sequences[:,1:],
        "b (h w c) -> b c h w", h = 32, w = 32, c = 3,
    ).float() / 255


def sample_wandb_grid(model, num_samples):
    samples = naive_sample(model, num_samples)
    image_grid = torchvision.utils.make_grid(samples)
    images = wandb.Image(image_grid)
    return images


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
        project = "ssm-image-generation-test",
        notes = "testing mamba image generation",
        tags = ["ssm", "cifar"],
    )

    wandb.log({"samples": samples})
