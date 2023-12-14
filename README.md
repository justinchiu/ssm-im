# ssm-im
small experiments with AR image modeling with Mamba.

## setup
install with `poetry lock && poetry install` or use conda and install things by hand.
mostly just the standard torch stuff + wandb.

## run
Current best:
`python main.py --batch-size 64 --lr 1e-3 --n-layer 16 --d-model 512`
Likely: setting --clip-grad-val 1 will work better.

## experiments
logged in [wandb](https://wandb.ai/chiu-justin-t/ssm-cifar-tokenized).
