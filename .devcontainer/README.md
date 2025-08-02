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

> Warning: Dev Containers (aka `devcontainers`) is an experimental feature and we are not testing in CI. Please submit any feedback using the issues on GitHub.

## Prerequisites
- [Docker](https://docs.docker.com/get-started/get-docker/) installed and configured on your host system
- IDEs: Both VS Code and Cursor have Dev Containers extensions
- Appropriate NVIDIA drivers (compatible with CUDA 12.8)
- For models that require authentication, set your Hugging Face token env var `HF_TOKEN` in your local startup (.bashrc, .zshrc or .profile file). Many models do not require this token.

## Quick Start

### There are two ways to build the development container image:

#### Method 1: Slow Build (better-tested, more reliable)
Build from your current source code:

```bash
./container/build.sh --target local-dev
```

The container will be built and give certain file permissions to your local uid and gid.

> Note: Currently local-dev is only implemented for --framework VLLM

#### Method 2: Fast Build (Depends on CI reliability)
Use a pre-built CI image from GitLab registry:

1. **Find your current Git SHA:**
   ```bash
   git rev-parse HEAD
   ```

2. **Search the GitLab registry -- you must be on the company VPN:**
   Go to https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/container_registry/85325

3. **Find your SHA:** Search for your commit SHA (e.g., `e61f1c8a40dafa7780d4983e545cc1eb09e9d2ee`)

4. **Copy the full container URL.** Hover over the clipboard icon to get the complete URL (e.g., `gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:e61f1c8a40dafa7780d4983e545cc1eb09e9d2ee-29598585-vllm-amd64`)

5. **Build the dev image:**
Below is an illustration, using the SHA you found in step 3.
   ```bash
   .devcontainer/build-dev-image.sh gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:e61f1c8a40dafa7780d4983e545cc1eb09e9d2ee-29598585-vllm-amd64
   ```

   This creates a development image called `dynamo:latest-vllm-local-dev`

### Install Dev Containers extension in your IDE

**VS Code:**
- Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) from Microsoft marketplace

**Cursor:**
- Press `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X` (Linux/Windows) to open Extensions
- Search for "Dev Containers" and install the one by **Anysphere** (Do not download the version from Microsoft as it is not compatible with Cursor)

Now open Dynamo folder in your IDE:
- Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Linux/Windows)
- Select "Dev Containers: Open Folder in Container"

Optional: If you want to mount your Hugging Face cache, go to `.devcontainer` and uncomment in the mounts section:

```json
// "source=${localEnv:HF_HOME},target=/home/ubuntu/.cache/huggingface,type=bind", // Uncomment to enable HF Cache Mount. Make sure to set HF_HOME env var in you .bashrc
```
Make sure HF_HOME is sourced in your .bashrc or .zshenv and your vscode default terminal is set properly.

Now wait for Initialization:
- The container will mount your local code
- `post-create.sh` will build the project and configure the environment

If `post-create.sh` fails, you can try to debug or [submit](https://github.com/ai-dynamo/dynamo/issues) an issue on GitHub.

## Development Flow

If you make changes to Rust code and want to compile, use [cargo build](https://doc.rust-lang.org/cargo/commands/cargo-build.html). This will update Rust binaries such as dynamo-run.

```bash
cd /home/ubuntu/dynamo && cargo build --locked --profile dev
```

Note that the cargo target directory is defined in devcontainer.json in `rust-analyzer.cargo.targetDir`, which should be `/home/ubuntu/dynamo/.build/target`. You can verify this by typing:
```bash
$ cargo metadata --format-version=1 | jq -r '.target_directory'
/home/ubuntu/dynamo/.build/target
```

Before pushing code to GitHub, remember to run `cargo fmt` and `cargo clippy`

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
- Edits to files are propogated to local repo due to the volume mount
- SSH and GPG agent passthrough orchestrated by devcontainer

File Structure:
- Local dynamo repo mounts to `/home/ubuntu/dynamo`
- Python venv in `/opt/dynamo/venv`
- Build artifacts in `dynamo/.build/target`
- Hugging Face cache preserved between sessions (either mounting your host .cache to the container, or your `HF_HOME` to `/home/ubuntu/.cache/huggingface`)
- Bash memory preserved between sessions at `/home/ubuntu/.commandhistory` using docker volume `dynamo-bashhistory`
- Precommit peeserved between sessions at `/home/ubuntu/.cache/precommit` using docker volume `dynamo-precommit-cache`

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

### SSH Keys for Git Operations

SSH keys need to be loaded in your SSH agent to work properly in the container. Can check out [VSCode docs](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials) for more details.

```bash
# In devcontainer, Check if your keys are loaded in the agent
ssh-add -l

# On local host, if your key isn't listed, add it
eval "$(ssh-agent)"  # Start the agent if not running
ssh-add ~/.ssh/id_rsa
```

Verify access by running `ssh -T git@github.com` in both host and container.

## Troubleshooting

### Environment Variables Not Set in Container?

If your environment variables are not being set in your devcontainer (e.g., `echo $HF_TOKEN` returns empty), and these variables are defined in your `~/.bashrc`, there are two ways to ensure they are properly sourced:

1. Add `source ~/.bashrc` to your `~/.bash_profile`, OR
2. Add `source ~/.bashrc` to your `~/.profile` AND ensure `~/.bash_profile` does not exist

Note: If both `~/.bash_profile` and `~/.profile` exist, bash will only read `~/.bash_profile` for login shells. Therefore, if you choose option 2, you must remove or rename `~/.bash_profile` to ensure `~/.profile` (and consequently `~/.bashrc`) is sourced.


See VS Code Dev Containers [documentation](https://code.visualstudio.com/docs/devcontainers/containers) for more details.

### Volume Corruption Issues

If you encounter strange errors (like `postCreateCommand` failing with exit code 1), your Docker volumes may be corrupted.

**Solution: Wipe Docker Volumes**

```bash
# Remove Dynamo volumes that are specified in devcontainer.json (may be corrupted)
docker volume rm dynamo-bashhistory dynamo-precommit-cache

# Or remove all volumes (use with caution)
docker volume prune
```

**Note:** This resets bash history and pre-commit cache, but preserves your source code.

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

### Build Issues

If you encounter build errors or strange compilation issues, try running `cargo clean` to clear the build cache and rebuild from scratch.

If `cargo clean` doesn't resolve the issue, you can manually remove the build directory outside the container:

```bash
# Replace <user> with your actual username
sudo rm -rf $HOME/dynamo/.build/target
```

This forces a complete rebuild of all Rust artifacts.
