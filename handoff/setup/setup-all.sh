#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==1/4 prereqs=="
bash "$HERE/setup-prereqs.sh"

echo
echo "==2/4 GitHub=="
bash "$HERE/setup-github.sh"

echo
echo "==3/4 Google Cloud=="
bash "$HERE/setup-gcloud.sh"

echo
echo "==4/4 Hugging Face=="
bash "$HERE/setup-huggingface.sh"

echo
echo "Setup complete. Next:"
echo "  bash handoff/unpack-state.sh"
