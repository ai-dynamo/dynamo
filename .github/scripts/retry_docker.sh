# Retry docker operations with exponential backoff.
# Safe under `set -e`: the `if` conditional context prevents a failed
# `docker <operation>` from triggering an immediate exit.
retry_docker_operation() {
  local operation="$1"
  local image="$2"
  local max_attempts=3
  local wait_seconds=10
  local attempt=1

  if [[ "$operation" != "push" && "$operation" != "pull" ]]; then
    echo "Unsupported docker operation: $operation (expected: push|pull)" >&2
    return 2
  fi

  while true; do
    if docker "$operation" "$image"; then
      return 0
    fi
    echo "Docker ${operation} failed for $image (attempt ${attempt}/${max_attempts})." >&2

    if (( attempt >= max_attempts )); then
      echo "Docker ${operation} failed after ${max_attempts} attempts: $image" >&2
      return 1
    fi

    echo "Retrying docker ${operation} in ${wait_seconds}s..."
    sleep "$wait_seconds"
    attempt=$((attempt + 1))
    wait_seconds=$((wait_seconds * 2))
    if (( wait_seconds > 120 )); then
      wait_seconds=120
    fi
  done
}

retry_push() {
  local image="$1"
  retry_docker_operation push "$image"
}

retry_pull() {
  local image="$1"
  retry_docker_operation pull "$image"
}
