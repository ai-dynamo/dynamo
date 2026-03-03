# Retry docker push with exponential backoff.
# Safe under `set -e`: the `if` conditional context prevents a failed
# `docker push` from triggering an immediate exit.
retry_push() {
  local image="$1"
  local max_attempts=3
  local wait_seconds=10
  local attempt=1

  while true; do
    if docker push "$image"; then
      return 0
    fi
    echo "Push failed for $image (attempt ${attempt}/${max_attempts})." >&2

    if (( attempt >= max_attempts )); then
      echo "Push failed after ${max_attempts} attempts: $image" >&2
      return 1
    fi

    echo "Retrying in ${wait_seconds}s..."
    sleep "$wait_seconds"
    attempt=$((attempt + 1))
    wait_seconds=$((wait_seconds * 2))
    if (( wait_seconds > 120 )); then
      wait_seconds=120
    fi
  done
}
