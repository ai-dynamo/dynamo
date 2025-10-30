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

## Framework-Specific Devcontainers

This directory contains framework-specific devcontainer configurations generated from a Jinja2 template:

- **`vllm/`** - Development environment for vLLM framework
- **`sglang/`** - Development environment for SGLang framework
- **`trtllm/`** - Development environment for TensorRT-LLM framework

### Template System

The devcontainer configurations are generated from:
- **`devcontainer.json.j2`** - Jinja2 template with framework variables
- **`gen_devcontainer_json.py`** - Python script to generate configs

To regenerate the framework-specific configurations after making changes:
```bash
cd .devcontainer
python3 gen_devcontainer_json.py
```

**Important**: Do not edit the generated `devcontainer.json` files directly. They contain auto-generated warnings and will be overwritten. Instead, edit the `devcontainer.json.j2` template and regenerate.

#### Why We Use Templates Instead of a Single devcontainer.json

The Dev Container Extension requires that each `devcontainer.json` file follow a specific directory convention, which results in significant duplication across framework-specific configurations. See https://code.visualstudio.com/remote/advancedcontainers/connect-multiple-containers

```
📁 project-root
    📁 .git
    📁 .devcontainer
      📁 python-container
        📄 devcontainer.json
      📁 node-container
        📄 devcontainer.json
    📁 python-src
        📄 hello.py
    📁 node-src
        📄 hello.js
    📄 docker-compose.yml
```

Alternative approaches using undocumented methods (e.g., changing devcontainer.json name, custom configurations) were explored but proved unsuccessful. The template system was developed to minimize duplication while maintaining compatibility with the Dev Container Extension's directory requirements.

Until the Microsoft Dev Container Extension adds new functionalities, this remains the recommended approach for managing multiple Dev Container configurations.

```mermaid
graph TB
    subgraph "Developer Laptop"
        subgraph IDE["Cursor/VS Code"]
            EXT["Dev Container Extension"]
            SSHEXT["Remote-SSH Extension"]
        end
        TERM["iTerm/Terminal"]
    end

    subgraph WS["Linux Workstation"]
        DIR["~/dynamo<br/>(host directory)"]
        HFCACHE["~/.cache/huggingface<br/>(host cache)"]
        GITCONFIG["~/.gitconfig<br/>(host git config)"]

        subgraph CONTAINER["Docker Container<br/>vsc-dynamo-SHA-uid<br/>Running as: ubuntu user"]
            MOUNT["/home/ubuntu/dynamo<br/>(mounted directory)"]
            HFMOUNT["/home/ubuntu/.cache/huggingface<br/>(mounted cache)"]
            GITCONFIGCOPY["/home/ubuntu/.gitconfig<br/>(via Dev Container setting)"]
            TOOLS["rust-analyzer<br/>cargo<br/>etc."]
        end

        IMAGE["Docker Image<br/>dynamo:latest-{framework}-local-dev<br/>(vllm/sglang/trtllm)"]

        IMAGE -->|"docker run<br/>as ubuntu user"| CONTAINER
    end

    TERM -->|"SSH Connection"| DIR
    SSHEXT -->|"1. Remote-SSH"| DIR
    EXT -->|"2. Dev Container:<br/>Open Folder (via ssh)"| IMAGE
    EXT -->|"3. IDE/shell via<br/>ssh+docker exec"| CONTAINER
    DIR <-.->|"Volume Mount<br/>(bidirectional)"| MOUNT
    HFCACHE <-.->|"Volume Mount<br/>(HF cache)"| HFMOUNT
    GITCONFIG -->|"Copy"| GITCONFIGCOPY

    style IDE fill:#e1f5fe
    style EXT fill:#cfe2ff
    style SSHEXT fill:#cfe2ff
    style TERM fill:#e1f5fe
    style DIR fill:#fff3e0
    style HFCACHE fill:#fff3e0
    style GITCONFIG fill:#fff3e0
    style IMAGE fill:#f3e5f5
    style CONTAINER fill:#d4edda
    style MOUNT fill:#fff9c4
    style HFMOUNT fill:#fff9c4
    style GITCONFIGCOPY fill:#fff9c4
    style TOOLS fill:#ffebee
```

## Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-started/get-docker/) installed and configured on your host system
- IDEs: Use either the VS Code or Cursor. Both have Dev Containers extensions
- Appropriate NVIDIA drivers (compatible with CUDA 12.8+)
- For models that require authentication, set your Hugging Face token env var `HF_TOKEN` in your local startup (.profile or .zprofile file). Many public models do not require this token.

### Required Files and Directories

You must have the following path on your host.

- **`~/.cache/huggingface`**: This directory is mounted into the container for Hugging Face model caching.


## Steps to Get Started

Follow these steps to get your NVIDIA Dynamo development environment up and running:

### Step 0: Build the Development Container Image

Build the appropriate framework image (e.g., `dynamo:latest-vllm-local-dev`) from scratch from the source:

```bash
# Single command approach (recommended)
export FRAMEWORK=VLLM         # Note: any of VLLM, SGLANG, TRTLLM can be used
./container/build.sh --framework $FRAMEWORK --target local-dev

# Now you've created both dynamo:latest-vllm and dynamo:latest-vllm-local-dev
```

Alternatively, you can build a development container, then build local-dev:

```bash
export FRAMEWORK=VLLM

./container/build.sh --framework $FRAMEWORK
# Now you have a development image dynamo:latest-vllm

./container/build.sh --dev-image dynamo:latest-${FRAMEWORK,,}
# Now you have a local-dev image dynamo:latest-vllm-local-dev
```

The local-dev image will give you local user permissions matching your host user and includes extra developer utilities (debugging tools, text editors, system monitors, etc.).

### Step 1: Choose Your Framework

Select the appropriate devcontainer based on your framework:
- Use `vllm/devcontainer.json` for vLLM development
- Use `sglang/devcontainer.json` for SGLang development
- Use `trtllm/devcontainer.json` for TensorRT-LLM development

When opening the devcontainer in VS Code/Cursor, navigate to the specific framework directory (e.g., `.devcontainer/vllm/`) and open that devcontainer.json.

### Step 2: Install Dev Containers Extension

**For Cursor:**
- Press `Cmd+Shift+X` (Mac) or `Ctrl+Shift+X` (Linux/Windows) to open Extensions
- Search for "Dev Containers" and install the one by **Anysphere** (Do not download the version from Microsoft as it is not compatible with Cursor)

**For VS Code:**
- Install [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) from Microsoft marketplace


### Step 3: Launch the Development Environment

1. Open `dynamo` folder in your IDE
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Linux/Windows)
3. Select "Dev Containers: Open Folder in Container", select your `dynamo` folder and open.

### Step 4: Optional But Highly Recommended Setup

For the best development experience, we recommend configuring the following:

#### Git Configuration
- **Personal `.gitconfig`**: If not done already, make your own `~/.gitconfig` file with your name, email, and preferred Git settings
- **Dev Container Integration**: Enable the `Copy the .gitconfig file to the devcontainer` setting by going to IDE Settings → Extensions → Dev Containers → **☑ Copy the .gitconfig file to the devcontainer**

#### SSH Authentication
To enable seamless Git operations (push/pull) via SSH:

1. **Configure ssh-agent on your host**: Search online for "how to set up ssh-agent on Ubuntu on .profile" and ensure it's configured in your `~/.profile` so the ssh-agent runs before Cursor launches

2. **Enable SSH forwarding in IDE**:
   Dev Containers set up -> check **☑ Dev Containers: Enable SSHAgent Forwarding**

3. **Verify SSH forwarding works**:
   - The post-create.sh script will check and report SSH agent status
   - You should see "SSH agent forwarding is working" when the container starts

This setup allows you to use Git commands normally within the container without additional authentication steps.


### Step 5: Wait for Initialization

The container will automatically:
- Mount your local code to `/workspace`
- Run `post-create.sh` to build the project and configure the environment

If `post-create.sh` fails, you can try to debug or [submit](https://github.com/ai-dynamo/dynamo/issues) an issue on GitHub.

## Development Flow

### Building Rust Code

If you make changes to Rust code and want to compile, use [cargo build](https://doc.rust-lang.org/cargo/commands/cargo-build.html). This will update Rust binaries such as dynamo-run.

```bash
cd /workspace && cargo build --locked --profile dev
```

Verify that builds are in the pre-defined `target` directory:
```bash
$ cargo metadata --format-version=1 | jq -r '.target_directory'
/workspace/target  <-- this is the target path
```

If cargo is not installed and configured properly, you will see one or more errors, such as the following:
```
error: could not find `Cargo.toml` in $HOME or any parent directory
```

Lastly, before pushing code to GitHub, remember to run `cargo fmt` and `cargo clippy`

### Updating Python Bindings

If you make changes to Rust code and want to propagate to Python bindings then can use [maturin](https://www.maturin.rs/#usage) (pre-installed). This will update the Python bindings with your new Rust changes.

```bash
cd /workspace/lib/bindings/python && maturin develop
```

## What's Inside
Development Environment:
- Rust and Python toolchains
- GPU acceleration
- VS Code or Cursor extensions for Rust and Python
- Persistent build cache in `target/` directory enables fast incremental builds (only changed files are recompiled) via `cargo build --locked --profile dev`
- Edits to files are propagated to local repo due to the volume mount
- SSH and GPG agent passthrough orchestrated by devcontainer

File Structure:
- Local dynamo repo mounts to `/workspace`
- Python venv in `/opt/dynamo/venv`
- Build artifacts in `/workspace/target`
- Hugging Face cache preserved between sessions (either mounting your host .cache to the container, or your `HF_HOME` to `/home/ubuntu/.cache/huggingface`)
- Bash memory preserved between sessions at `/home/ubuntu/.commandhistory` using docker volume `dynamo-bashhistory`
- Precommit preserved between sessions at `/home/ubuntu/.cache/precommit` using docker volume `dynamo-precommit-cache`

## Documentation

To look at the docs run:
```bash
cd /workspace/target/doc && python3 -m http.server 8000
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

You can create a custom devcontainer configuration by copying the main configuration to another directory inside the `.devcontainer` directory. Below is an example where the custom name is `jensen_dev`, but feel free to name the directory whatever you want:

```bash
# By convention, Dev Container will look at the project's .devcontainer/<path>/devcontainer.json file.
# Example: copy the main devcontainer configuration and then edit the new json file
mkdir -p .devcontainer/jensen_dev
cp .devcontainer/devcontainer.json .devcontainer/jensen_dev/devcontainer.json
```

Common customizations include additional mounts, environment variables, IDE extensions, and build arguments. When you open a new Dev Container, you can pick from any of the `.devcontainer/<path>/devcontainer.json` files available.


### SSH Keys for Git Operations

If you have ssh-agent running on the host, then `git push` should just work. If not, you may need to set up ssh-agent, or have SSH keys set up inside the container (more hassle).


## Troubleshooting

### Environment Variables Not Set in Container?

Dev containers have limited access to host environment variables for security reasons. Here's how to properly pass environment variables:

#### Method 1: Use devcontainer.json
Add environment variables to `.devcontainer/devcontainer.json`:
```json
{
  "remoteEnv": {
    "HF_TOKEN": "${localEnv:HF_TOKEN}",
    "GITHUB_TOKEN": "${localEnv:GITHUB_TOKEN}",
    "SSH_AUTH_SOCK": "${env:SSH_AUTH_SOCK}"
  }
}
```

#### Method 2: Host Shell Configuration
In order to ensure your host environment variables are available to the Dev Containers:

**For bash users:**
- Put variables in `~/.profile` (for login shells)
- Ensure `~/.bash_profile` sources `~/.bashrc` if both exist
- Some users report that variables in `~/.bashrc` aren't picked up by dev containers, so try moving them to `~/.profile` to see if that solves the problem

**For zsh users:**
- Put variables in `~/.zprofile` (for login shells)
- Some users report that variables in `~/.zshrc` aren't picked up by dev containers, so try moving them to `~/.zprofile` to see if that solves the problem

#### Method 3: Environment File
Create a `.env` file in your project root and reference it in devcontainer.json:

**Example .env file:**
```bash
# API Tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Development settings
DEBUG=true
LOG_LEVEL=debug

# Custom paths
CUSTOM_MODEL_PATH=/path/to/models
```

**Reference in devcontainer.json:**
```json
{
  "remoteEnv": {
    "ENV_FILE": "${localWorkspaceFolder}/.env"
  }
}
```

See VS Code Dev Containers [documentation](https://code.visualstudio.com/docs/devcontainers/containers) for more details.

### Build Issues

If you encounter build errors or strange compilation issues, try running `cargo clean`, then rebuild from scratch.

If `cargo clean` doesn't resolve the issue, it is possible that some of the files were created by root (using the `run.sh` script). You can manually remove the build target by going to your host (outside the container), and remove the target:

```bash
sudo rm -rf <your dynamo path on the host machine>/target
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

1. **Missing a local-dev image:**
   ```bash
   # Check if the required local-dev image exists
   docker images | grep dynamo

   # If missing, build the dev image first, then build local-dev
   export FRAMEWORK=VLLM  # Replace with VLLM, SGLANG, or TRTLLM
   ./container/build.sh --framework $FRAMEWORK
   # change to lower case portable way across shells
   ./container/build.sh --dev-image dynamo:latest-$(echo "$FRAMEWORK" | tr '[:upper:]' '[:lower:]') --framework "$FRAMEWORK"
   # Now you have dynamo:latest-vllm-local-dev
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
  Then rebuild without cache. In your IDE:
  *Dev Containers: Rebuild Without Cache and Reopen in Container*

### devcontainer.json Changes Not Being Picked Up

If you've made changes to `devcontainer.json`, `post-create.sh`, or other devcontainer-related files but they're not being applied when you rebuild the container, the changes may be cached.

**Solution: Force Devcontainer Rebuild**

1. **Rebuild Container (Recommended):**
   In your IDE Command Palette (Ctrl+Shift+P):
   *Dev Containers: Rebuild Container*

2. **If that doesn't work, rebuild without cache:**
   In your IDE Command Palette (Ctrl+Shift+P):
   *Dev Containers: Rebuild Without Cache and Reopen Container*

3. **For persistent issues, manually remove the devcontainer image:**
   ```bash
   # List devcontainer images
   docker images | grep devcontainer

   # And remove all IDE devcontainer images (more thorough)
   docker images | grep "^vsc-" | awk '{print $3}' | xargs docker rmi

   # Then rebuild in your IDE
   Dev Containers: Rebuild Container
   ```

**Note:** The "Rebuild Container Without Cache and Reopen Container" option is the most thorough and will ensure all your changes are applied, but it takes longer as it rebuilds everything from scratch.
