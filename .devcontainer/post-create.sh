#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

# Ensure we're not running as root
if [ "$(id -u)" -eq 0 ]; then
    echo "❌ ERROR: This script should not be run as root!"
    echo "The script should run as the 'ubuntu' user, not root."
    echo "Current user: $(whoami) (UID: $(id -u))"
    exit 1
fi

# Verify we're running as the expected user
if [ "$(whoami)" != "ubuntu" ]; then
    echo "⚠️  WARNING: Expected to run as 'ubuntu' user, but running as '$(whoami)'"
    echo "This might cause permission issues."
fi

echo "Running post-create script as user: $(whoami) (UID: $(id -u))"

trap 'echo "❌ ERROR: Command failed at line $LINENO: $BASH_COMMAND"; echo "⚠️ This was unexpected and setup was not completed. Can try to resolve yourself and then manually run the rest of the commands in this file or file a bug."' ERR

retry() {
    # retries for connectivity issues in installs
    local retries=3
    local count=0
    until "$@"; do
        exit_code=$?
        wait_time=$((2 ** count))
        echo "Command failed with exit code $exit_code. Retrying in $wait_time seconds..."
        sleep $wait_time
        count=$((count + 1))
        if [ $count -ge $retries ]; then
            echo "Command failed after $retries attempts."
            return $exit_code
        fi
    done
    return 0
}

set -x

# Pre-commit hooks
cd $HOME/dynamo && pre-commit install && retry pre-commit install-hooks
pre-commit run --all-files || true # don't fail the build if pre-commit hooks fail

# Set build directory
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/dynamo/.build/target}
mkdir -p $CARGO_TARGET_DIR

uv pip uninstall --yes ai-dynamo ai-dynamo-runtime 2>/dev/null || true

# Build project, with `dev` profile it will be saved at $CARGO_TARGET_DIR/debug
cargo build --locked --profile dev --features mistralrs

# install the python bindings
(cd $HOME/dynamo/lib/bindings/python && retry maturin develop)

# installs overall python packages, grabs binaries from .build/target/debug
cd $HOME/dynamo && retry env DYNAMO_BIN_PATH=$CARGO_TARGET_DIR/debug uv pip install -e .

# Extract the PYTHONPATH line from README.md
PYTHONPATH_LINE=$(grep "^export PYTHONPATH=" $DYNAMO_HOME/README.md | head -n1)
if [ -n "$PYTHONPATH_LINE" ]; then
    # Remove the ${PYTHONPATH}: prefix if it exists, then replace $(pwd) with the actual path
    MODIFIED_LINE=$(echo "$PYTHONPATH_LINE" | sed 's/\${PYTHONPATH}://g' | sed "s|\$(pwd)|$DYNAMO_HOME|g")
    eval "$MODIFIED_LINE"
    # Also add to .bashrc for persistence (with expanded path)
    if ! grep -q "export PYTHONPATH=" ~/.bashrc; then
        # MODIFIED_LINE already has $DYNAMO_HOME expanded to /home/ubuntu/dynamo
        echo "$MODIFIED_LINE" >> ~/.bashrc
    fi
else
    # Back-up version if README.md changed. This is the version from 2025-08-19
    export PYTHONPATH=$DYNAMO_HOME/components/frontend/src:$DYNAMO_HOME/components/planner/src:$DYNAMO_HOME/components/backends/vllm/src:$DYNAMO_HOME/components/backends/sglang/src:$DYNAMO_HOME/components/backends/trtllm/src:$DYNAMO_HOME/components/backends/llama_cpp/src:$DYNAMO_HOME/components/backends/mocker/src
fi

if ! grep -q "export GPG_TTY=" ~/.bashrc; then
    echo "export GPG_TTY=$(tty)" >> ~/.bashrc
fi

# Unset empty tokens/variables to avoid issues with authentication and SSH
if ! grep -q "# Unset empty tokens" ~/.bashrc; then
    echo -e "\n# Unset empty tokens and environment variables" >> ~/.bashrc
    echo '[ -z "$HF_TOKEN" ] && unset HF_TOKEN' >> ~/.bashrc
    echo '[ -z "$GITHUB_TOKEN" ] && unset GITHUB_TOKEN' >> ~/.bashrc
    echo '[ -z "$SSH_AUTH_SOCK" ] && unset SSH_AUTH_SOCK' >> ~/.bashrc
fi

$HOME/dynamo/deploy/dynamo_check.py --import-check-only

{ set +x; } 2>/dev/null

# Check SSH agent forwarding status
if [ -n "$SSH_AUTH_SOCK" ]; then
    if ssh-add -l > /dev/null 2>&1; then
        echo "SSH agent forwarding is working - found $(ssh-add -l | wc -l) key(s):"
        ssh-add -l
    else
        echo "⚠️ SSH_AUTH_SOCK is set but ssh-add failed - agent may not be accessible"
    fi
else
    echo "⚠️ SSH agent forwarding not configured - SSH_AUTH_SOCK is not set"
fi

cat <<EOF

✅ SUCCESS: Built cargo project, installed Python bindings, configured pre-commit hooks

Example commands:
  cargo build --locked --profile dev              # Build Rust project in $CARGO_TARGET_DIR
  cd lib/bindings/python && maturin develop --uv  # Update Python bindings (if you changed them)
  cargo fmt && cargo clippy                       # Format and lint code before committing
  cargo doc --no-deps                             # Generate documentation
  uv pip install -e .                             # Install various Python packages Dynamo depends on
EOF
