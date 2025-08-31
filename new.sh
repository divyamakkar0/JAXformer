#!/bin/bash

source .env
SESSION="mysession"
# command="python 2axismain.p:wqy "
command="python test.py --checkpoint_steps=10 --n_device_axis 8 2 2 --name newEmbed4 --train_batch_size 192 --use_cache"
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

echo "Setting up remote tmux sessions..."
for i in $(seq 0 7); do
    ssh divyamakkar@${IPS[$i]} << EOF &
        tmux kill-session -t ${SESSION}_remote 2>/dev/null || true
        tmux new-session -d -s ${SESSION}_remote
        tmux send-keys -t ${SESSION}_remote "cd ~/Jaxformer && rm -rf samples && mkdir samples" C-m
        tmux send-keys -t ${SESSION}_remote "git fetch origin && git reset --hard origin/main" C-m
        tmux send-keys -t ${SESSION}_remote "bash setupTpu.sh" C-m
        tmux send-keys -t ${SESSION}_remote "wandb login $WANDB_KEY" C-m
        tmux send-keys -t ${SESSION}_remote "$command" C-m
        tmux new-window -t ${SESSION}_remote -n "monitor"
        tmux send-keys -t ${SESSION}_remote:monitor "watch -n 1 tpu-info" C-m
        tmux select-window -t ${SESSION}_remote:0
EOF
done
wait

echo "Remote sessions created. Creating local tmux for monitoring..."

# Create local tmux session for monitoring
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "main" \; \
  split-window -v \; split-window -v \; split-window -v \; \
  select-pane -t "$SESSION":0.0 \; split-window -h \; \
  select-pane -t "$SESSION":0.1 \; split-window -h \; \
  select-pane -t "$SESSION":0.2 \; split-window -h \; \
  select-pane -t "$SESSION":0.3 \; split-window -h \; \
  select-layout tiled

# Connect to remote tmux sessions
for i in $(seq 0 7); do
  tmux send-keys -t "$SESSION":0.$i "ssh divyamakkar@${IPS[$i]} -t tmux attach -t ${SESSION}_remote" C-m
done

echo "Local tmux session created. Attaching..."
tmux attach -t "$SESSION"