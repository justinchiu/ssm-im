# ssm-im
small experiments with AR image modeling with Mamba.

## setup
install with `poetry lock && poetry install` or use conda and install things by hand.
mostly just the standard torch stuff + wandb.

## run
### without argparser
`python main.py --num_epochs 50 --n_layer 8 --batch_size 64` for the most recent.

### with argparser
`python main.py --batch-size 128 --num-workers 4 --d-model 256 --n-layer 4 --lr 0.001 --num-epochs 5 --grad-accumulation-steps 1 --save-model-steps 100` for most recent

## experiments
logged in [wandb](https://wandb.ai/chiu-justin-t/ssm-cifar-tokenized).
