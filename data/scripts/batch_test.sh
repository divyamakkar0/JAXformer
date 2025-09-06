set -euo pipefail

# TODO: fill in these variables
PY_SCRIPT="" 
RUNTIME_S="5" 
RESULTS="bs_benchmark_$(date +%Y%m%d_%H%M%S).csv"
EXTRA_ARGS=""

echo "batch_size,tokens_per_s" > "$RESULTS"

for bs in $(seq 25 50 750) 750; do
  if grep -q "^$bs," "$RESULTS"; then
    continue
  fi

  echo "Testing BATCH_SIZE=$bs ..."

  LOG="$(BENCHMARK=1 BATCH_SIZE="$bs" timeout --signal=INT "${RUNTIME_S}s" \
        python "$PY_SCRIPT" $EXTRA_ARGS 2>&1 || true)"

  RATE="$(printf "%s\n" "$LOG" \
    | sed -nE 's/.*([0-9]+(\.[0-9]+)?) ?tokens\/s.*/\1/p' \
    | tail -n1)"
  [ -z "${RATE:-}" ] && RATE="NA"


  echo "$LOG"

  if [ -z "${RATE:-}" ]; then
    RATE="NA"
  fi

  echo "$bs,$RATE" | tee -a "$RESULTS"
done

echo "Done. Results saved to $RESULTS"
