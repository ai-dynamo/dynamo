#!/bin/bash
set -e

# Create and activate Python virtual environment
python3 -m venv venv
. ./venv/bin/activate

# Build Rust components first
cargo build --release

# Install Dynamo with all dependencies
pip install -e .[all]

# Install development tools
pip install pytest isort mypy pylint

echo "Development environment setup complete!" 