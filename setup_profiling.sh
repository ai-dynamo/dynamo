#!/usr/bin/env bash
# Install profiling dependencies

set -euo pipefail

apt-get update -qq
apt-get install -y -qq heaptrack
pip3 install -q psutil aiohttp

if [[ ! -d "$HOME/FlameGraph" ]]; then
    git clone --depth 1 -q https://github.com/brendangregg/FlameGraph.git "$HOME/FlameGraph"
fi

echo "Done."
