# Retry docker push/pull with exponential backoff.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retry.sh"

retry_push() {
  retry docker push "$1"
}

retry_pull() {
  retry docker pull "$1"
}
