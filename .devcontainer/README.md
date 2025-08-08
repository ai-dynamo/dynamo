<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NVIDIA Dynamo Development Environment

> Warning: Dev Containers (aka `devcontainers`) is an evolving feature and we are not testing in CI. Please submit any problem/feedback using the issues on GitHub.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-started/get-docker/) installed and configured on your host system
- IDEs: Use either the VS Code or Cursor. Both have Dev Containers extensions
- Appropriate NVIDIA drivers (compatible with CUDA 12.8+)
- For models that require authentication, set your Hugging Face token env var `HF_TOKEN` in your local startup (.bashrc, .zshrc or .profile file). Many public models do not require this token.

### Required Files and Directories

The following files and directories are required on your host system for the devcontainer to work properly:

- **`.gitconfig`**: Must exist in your home directory (`~/.gitconfig`). This file is mounted into the container for Git configuration.
- **`~/.cache/huggingface`**: This directory is mounted into the container for Hugging Face model caching. If it doesn't exist, it will be created automatically.

If these files/directories are missing, you may encounter Docker mount errors when starting the devcontainer.

## Quick Start Guide

Follow these steps to get your NVIDIA Dynamo development environment up and running:

### Step 1: Build the Development Container Image

Build `dynamo:latest-vllm-local-dev` from scratch from the source:
- Note that currently, `local-dev` is only implemented for --framework VLLM

```bash
./container/build.sh --target local-dev
```

The container will be built and give certain file permissions to your local uid and gid.

### Step 2: Install Dev Containers Extension

**For VS Code:**
- Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) from Microsoft marketplace

**For Cursor:**
- Press `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X` (Linux/Windows) to open Extensions
- Search for "Dev Containers" and install the one by **Anysphere** (Do not download the version from Microsoft as it is not compatible with Cursor)

### Step 3: Launch the Development Environment

1. Open `dynamo` folder in your IDE
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Linux/Windows)
3. Select "Dev Containers: Open Folder in Container", select your `dynamo` folder and open.

### Step 4: Optional - Mount Hugging Face Cache

If you want to mount your Hugging Face cache for faster model loading:

1. Go to `.devcontainer` folder
2. Add this line in the mounts section:
   ```json
   "source=${localEnv:HF_HOME},target=/home/ubuntu/.cache/huggingface,type=bind", // Uncomment to enable HF Cache Mount. Make sure to set HF_HOME env var in you .bashrc
   ```
3. Make sure HF_HOME is sourced in your .bashrc or .zshenv and your IDE default terminal is set properly

### Step 5: Wait for Initialization

The container will automatically:
- Mount your local code to `/home/ubuntu/dynamo`
- Run `post-create.sh` to build the project and configure the environment

If `post-create.sh` fails, you can try to debug or [submit](https://github.com/ai-dynamo/dynamo/issues) an issue on GitHub.

## Development Flow

### Building Rust Code

If you make changes to Rust code and want to compile, use [cargo build](https://doc.rust-lang.org/cargo/commands/cargo-build.html). This will update Rust binaries such as dynamo-run.

```bash
cd /home/ubuntu/dynamo && cargo build --locked --profile dev
```

Verify that builds are in the pre-defined `dynamo/.build/target` and not `dynamo/workspace`:
```bash
$ cargo metadata --format-version=1 | jq -r '.target_directory'
/home/ubuntu/dynamo/.build/target
```

Before pushing code to GitHub, remember to run `cargo fmt` and `cargo clippy`

### Updating Python Bindings

If you make changes to Rust code and want to propagate to Python bindings then can use [maturin](https://www.maturin.rs/#usage) (pre-installed). This will update the Python bindings with your new Rust changes.

```bash
cd /home/ubuntu/dynamo/lib/bindings/python && maturin develop
```

## What's Inside
Development Environment:
- Rust and Python toolchains
- GPU acceleration
- VS Code or Cursor extensions for Rust and Python
- Persistent build cache in `.build/` directory enables fast incremental builds (only changed files are recompiled) via `cargo build --locked --profile dev`
- Edits to files are propagated to local repo due to the volume mount
- SSH and GPG agent passthrough orchestrated by devcontainer

File Structure:
- Local dynamo repo mounts to `/home/ubuntu/dynamo`
- Python venv in `/opt/dynamo/venv`
- Build artifacts in `dynamo/.build/target`
- Hugging Face cache preserved between sessions (either mounting your host .cache to the container, or your `HF_HOME` to `/home/ubuntu/.cache/huggingface`)
- Bash memory preserved between sessions at `/home/ubuntu/.commandhistory` using docker volume `dynamo-bashhistory`
- Precommit preserved between sessions at `/home/ubuntu/.cache/precommit` using docker volume `dynamo-precommit-cache`

## Customization
Edit `.devcontainer/devcontainer.json` to modify:
- VS Code settings and extensions
- Environment variables
- Container configuration
- Custom Mounts

## Documentation

To look at the docs run:
```bash
cd ~/dynamo/.build/target/doc && python3 -m http.server 8000
```

VSCode will automatically port-forward and you can check them out in your browser.

## FAQ

### GPG Keys for Signing Git Commits
Signing commits using GPG should work out of the box according to [VSCode docs](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials#_sharing-gpg-keys).

If you run into version compatibility issues you can try:

```bash
# On Host
gpg --list-secret-keys
gpg --export-secret-keys --armor YOUR_KEY_ID > /tmp/key.asc

# In container
gpg1 --import /tmp/key.asc
git config --local gpg.program gpg1
```

> Warning: Switching local gpg to gpg1 can have ramifications when you are not in the container any longer.

### Custom devcontainer.json Configuration

You can create a custom devcontainer configuration by copying the main configuration to another directory:

```bash
# Copy the main devcontainer configuration and then edit the new json file
mkdir -p .devcontainer/my_custom
cp .devcontainer/devcontainer.json .devcontainer/my_custom/devcontainer.json
```

Common customizations include additional mounts, environment variables, VS Code extensions, and build arguments. When you open a new Dev Container, you can pick from any of the `.devcontainer/*/devcontainer.json` files available.

### SSH Keys for Git Operations

To use `git push` commands in the devcontainer, you need SSH keys set up inside the container. This is not done automatically for security reasons - it's up to you to decide whether you want to copy your SSH keys from the host.

#### Manual SSH Key Setup
```bash
# Copy SSH keys from host to container (run from host)
docker cp ~/.ssh/id_rsa <container_id>:/home/ubuntu/.ssh/
docker cp ~/.ssh/id_rsa.pub <container_id>:/home/ubuntu/.ssh/

# Set proper permissions (run inside container)
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
```

Test the connection with: `ssh -T git@github.com`

## Troubleshooting

### Environment Variables Not Set in Container?

If your environment variables are not being set in your devcontainer (e.g., `echo $HF_TOKEN` returns empty), and these variables are defined in your `~/.bashrc`, there are two ways to ensure they are properly sourced:

1. Add `source ~/.bashrc` to your `~/.bash_profile`, OR
2. Add `source ~/.bashrc` to your `~/.profile` AND ensure `~/.bash_profile` does not exist

Note: If both `~/.bash_profile` and `~/.profile` exist, bash will only read `~/.bash_profile` for login shells. Therefore, if you choose option 2, you must remove or rename `~/.bash_profile` to ensure `~/.profile` (and consequently `~/.bashrc`) is sourced.

See VS Code Dev Containers [documentation](https://code.visualstudio.com/docs/devcontainers/containers) for more details.

### Build Issues

If you encounter build errors or strange compilation issues, try running `cargo clean` to clear the build cache and rebuild from scratch.

If `cargo clean` doesn't resolve the issue, it is possible that some of the files were created by root (using the `run.sh` script). You can manually remove the build target by going to your host (outside the container), and remove the target:

```bash
sudo rm -rf <your dynamo path on the host machine>/.build/target
```

### Volume Corruption Issues

If you encounter strange errors (like `postCreateCommand` failing with exit code 1), your Docker volumes may be corrupted.

**Solution: Wipe Docker Volumes**

```bash
# Remove Dynamo volumes that are specified in devcontainer.json (may be corrupted)
docker volume rm dynamo-bashhistory dynamo-precommit-cache

# Or remove all volumes (use with caution).
docker rm -f <your running container(s)>
docker volume prune -f
```

**Note:** This resets bash history and pre-commit cache.

**Volume Mounts in devcontainer.json:**
- `dynamo-bashhistory` → `/home/ubuntu/.commandhistory` (bash history)
- `dynamo-precommit-cache` → `/home/ubuntu/.cache/pre-commit` (pre-commit cache)

### Permission Issues

If you start experiencing permission problems (e.g., "Permission denied" errors), you may need to fix file ownership outside the container. This commonly happens when `container/run.sh` runs as root, creating files with root ownership:

```bash
# Replace <user> with your actual username
cd <your dynamo directory at your host machine (not docker)>
sudo chown -R <user>:<user> .
```

This fixes ownership when files are created with different user IDs between the host and container.

### Container Starts But Immediately Stops

If you see errors like "container is not running" or "An error occurred setting up the container" in the devcontainer logs, the container is starting but then crashing immediately.

**Common Causes and Solutions:**

1. **Missing base image:**
   ```bash
   # Check if the required image exists
   docker images | grep dynamo

   # If missing, build the dev image first
   ./container/build.sh --target local-dev
   ```

2. **Container startup failure:**
   ```bash
   # Check container logs for the specific error
   docker logs <container-id>

   # Or check all recent containers
   docker ps -a --filter "label=devcontainer.local_folder=$(pwd)"
   ```

3. **Resource issues:**
   ```bash
   # Check available system resources
   free -h
   df -h

   # Restart Docker daemon if needed
   sudo systemctl restart docker
   ```

4. **Clean slate approach:**
   ```bash
   # Remove all related containers and images
   docker ps -a --filter "label=devcontainer.local_folder=$(pwd)" -q | xargs docker rm -f
   docker images | grep "^vsc-" | awk '{print $3}' | xargs docker rmi
   ```
  Then rebuild without cache. In VS Code or Cursor command:
  *Dev Containers: Rebuild Without Cache and Reopen in Container*

### devcontainer.json Changes Not Being Picked Up

If you've made changes to `devcontainer.json`, `post-create.sh`, or other devcontainer-related files but they're not being applied when you rebuild the container, the changes may be cached.

**Solution: Force Devcontainer Rebuild**

1. **Rebuild Container (Recommended):**
   In VS Code or Cursor Command Palette (Ctrl+Shift+P):
   *Dev Containers: Rebuild Container*

2. **If that doesn't work, rebuild without cache:**
   In VS Code or Cursor Command Palette (Ctrl+Shift+P):
   *Dev Containers: Rebuild Without Cache and Reopen Container*

3. **For persistent issues, manually remove the devcontainer image:**
   ```bash
   # List devcontainer images
   docker images | grep devcontainer

   # And remove all VS Code devcontainer images (more thorough)
   docker images | grep "^vsc-" | awk '{print $3}' | xargs docker rmi

   # Then rebuild in VS Code
   Dev Containers: Rebuild Container
   ```

**Note:** The "Rebuild Container Without Cache and Reopen Container" option is the most thorough and will ensure all your changes are applied, but it takes longer as it rebuilds everything from scratch.
