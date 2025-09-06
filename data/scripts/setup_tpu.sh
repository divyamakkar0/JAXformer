set -euo pipefail
source .env 2>/dev/null || true

IPS=(
  "LIST GOES HERE"
)

HEAD_IP="${HEAD_IP:-${IPS[0]}}"

SSH_USER="${SSH_USER:-$USER}"
SSH_KEY="${SSH_KEY:-}" 

RAY_PORT="${RAY_PORT:-6379}"      # ray head port
WORKDIR="${WORKDIR:-~/jaxformer}" # remote project dir on each node
PYTHON="${PYTHON:-python3}"       # python on the remote machines
MAIN_SCRIPT="${MAIN_SCRIPT:-main_distributed.py}"
MAIN_ARGS="${MAIN_ARGS:-}"        # optional args for your script


export HEAD_IP RAY_PORT WORKDIR PYTHON SSH_USER SSH_KEY

echo "[1/3] Starting Ray on all nodes (head: $HEAD_IP, port: $RAY_PORT)..."
printf "%s\n" "${IPS[@]}" | xargs -n 1 -P 0 -I {} bash run.sh {}

echo "[2/3] Waiting for Ray cluster to become ready..."

TIMEOUT_SEC=120
EXPECTED_NODES=${#IPS[@]}
DEADLINE=$(( $(date +%s) + TIMEOUT_SEC ))

while :; do
  set +e
  NODES=$(
    ssh -o StrictHostKeyChecking=no ${SSH_KEY:+-i "$SSH_KEY"} "$SSH_USER@$HEAD_IP" \
      "bash -lc '$PYTHON - <<\"PY\"
import time, ray
ray.init(address=\"auto\", ignore_reinit_error=True, namespace=\"_probe_\")
print(len(ray.nodes()))
PY
'"
  )
  STATUS=$?
  set -e
  if [[ $STATUS -eq 0 && \"$NODES\" =~ ^[0-9]+$ && $NODES -ge $EXPECTED_NODES ]]; then
    echo "Ray is up with $NODES/$EXPECTED_NODES nodes."
    break
  fi
  if [[ $(date +%s) -ge $DEADLINE ]]; then
    echo "Timed out waiting for Ray cluster. Got $NODES/$EXPECTED_NODES nodes." >&2
    exit 1
  fi
  sleep 3
done

echo "[3/3] Launching training on the head node..."
ssh -o StrictHostKeyChecking=no ${SSH_KEY:+-i "$SSH_KEY"} "$SSH_USER@$HEAD_IP" "bash -lc '
  set -e
  cd \"$WORKDIR\"
  # If your script needs the Ray address, pass it explicitly:
  #   $PYTHON -u $MAIN_SCRIPT --address $HEAD_IP:$RAY_PORT $MAIN_ARGS
  # Otherwise, if it uses ray.init(\"auto\"), the below is fine:
  $PYTHON -u $MAIN_SCRIPT $MAIN_ARGS
'"

echo "Done."
