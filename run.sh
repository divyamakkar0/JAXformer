#!/bin/bash

IP=$1
SESSION="trainingRun"
command="python main.py --checkpoint_steps=75 --n_device_axis 8 2 2 --name moe1B --train_batch_size 32 --use_cache --wandb --eval_steps 10"

echo "Running on $IP"

ssh adityamakkar@$IP "

    tmux kill-session -t $SESSION
    tmux new-session -d -s $SESSION

    tmux send-keys -t $SESSION:0 'cd ~/Jaxformer && rm -rf samples && mkdir samples' C-m
    tmux send-keys -t $SESSION:0 'git fetch origin && git reset --hard origin/main' C-m
    tmux send-keys -t $SESSION:0 'bash setupTpu.sh' C-m
    tmux send-keys -t $SESSION:0 '$command' C-m
"
echo "done commands"
