retry_push() {
  local image="$1"
  local max_attempts=8
  local wait_seconds=10
  local attempt=1

  while true; do
    if docker push "$image"; then
      return 0
    fi

    if (( attempt >= max_attempts )); then
      echo "Push failed after ${max_attempts} attempts: $image"
      return 1
    fi

    echo "Push failed for $image (attempt ${attempt}/${max_attempts}); retrying in ${wait_seconds}s..."
    sleep "$wait_seconds"
    attempt=$((attempt + 1))
    wait_seconds=$((wait_seconds * 2))
    if (( wait_seconds > 120 )); then
      wait_seconds=120
    fi
  done
}
