#!/bin/bash

conda deactivate
tmux new-session -s "ssm" -n "root" "tmux source-file tmux/session"
