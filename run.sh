#!/bin/bash

IP=$1 
SESSION="mysession"
command="python test.py --checkpoint_steps=10 --n_device_axis 8 2 2 --name newEmbed4 --train_batch_size 192 --use_cache"
WANDB_KEY=""

echo "Running on $IP"

ssh -o StrictHostKeyChecking=no divyamakkar@$IP "
    
    tmux kill-session -t $SESSION 2>/dev/null || true
    tmux new-session -d -s $SESSION

    tmux send-keys -t $SESSION:0 'cd ~/Jaxformer && rm -rf samples && mkdir samples' C-m
    tmux send-keys -t $SESSION:0 'git fetch origin && git reset --hard origin/main' C-m
    tmux send-keys -t $SESSION:0 'bash setupTpu.sh' C-m
    tmux send-keys -t $SESSION:0 'wandb login $WANDB_KEY' C-m
    tmux send-keys -t $SESSION:0 '$command' C-m
"
echo "done commands"