#!/bin/bash

source .env
SESSION="mysession"
# command="python 2axismain.p:wqy "

 command="python test.py --checkpoint_steps=1 --n_device_axis 8 2 2 --name moeTestFinal --train_batch_size 16 --use_cache --wandb --eval_steps 1"

# command="python testCheckpoint.py"

IPS=(
    "35.186.25.28"
    "35.186.39.76"
    "107.167.173.215"
    "35.186.132.44"
    "35.186.24.134"
    "35.186.58.69"
    "35.186.134.160"
    "35.186.107.62"
)

tmux new-session -d -s "$SESSION" -n "main" \; \
  split-window -v \; split-window -v \; split-window -v \; \
  select-pane -t "$SESSION":0.0 \; split-window -h \; \
  select-pane -t "$SESSION":0.1 \; split-window -h \; \
  select-pane -t "$SESSION":0.2 \; split-window -h \; \
  select-pane -t "$SESSION":0.3 \; split-window -h \; \
  select-layout tiled

tmux new-window -t "$SESSION" -n "monitor" \; \
  split-window -v \; split-window -v \; split-window -v \; \
  select-pane -t "$SESSION":1.0\; split-window -h \; \
  select-pane -t "$SESSION":1.1 \; split-window -h \; \
  select-pane -t "$SESSION":1.2 \;  split-window -h \; \
  select-pane -t "$SESSION":1.3 \; split-window -h \; \
  select-layout tiled

for i in $(seq 0 7); do
  tmux send-keys -t "$SESSION":0.$i "ssh adityamakkar@${IPS[$i]}" C-m
  tmux send-keys -t "$SESSION":0.$i "cd ~/Jaxformer && rm -rf samples && mkdir samples" C-m
  tmux send-keys -t "$SESSION":0.$i "git fetch origin && git reset --hard origin/main" C-m
  tmux send-keys -t "$SESSION":0.$i "bash setupTpu.sh" C-m
  tmux send-keys -t "$SESSION":0.$i "wandb login $WANDB_KEY" C-m
  tmux send-keys -t "$SESSION":0.$i "$command" C-m
done

for i in $(seq 0 7); do
  tmux send-keys -t "$SESSION":1.$i "ssh adityamakkar@${IPS[$i]}" C-m
  tmux send-keys -t "$SESSION":1.$i "watch -n 1 tpu-info" C-m
done

tmux attach -t "$SESSION"
