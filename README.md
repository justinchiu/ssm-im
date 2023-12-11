# ssm-im
small experiments with AR image modeling with Mamba.

## setup
install with `poetry lock && poetry install` or use conda and install things by hand.
mostly just the standard torch stuff + wandb.

## run
`python main.py --batch-size 64 --num-workers 4 --d-model 256 --n-layer 8 --lr 0.001 --num-epochs 50 --grad-accumulation-steps 1 --save-model-steps 500`

## experiments
logged in [wandb](https://wandb.ai/chiu-justin-t/ssm-cifar-tokenized).
