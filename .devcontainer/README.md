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

> Warning: devcontainers is an experimental feature and we are not testing in CI. Please submit any feedback using the issues on github.

## Prerequisites
- [Docker](https://docs.docker.com/get-started/get-docker/) installed and configured on your host system
- Visual Studio Code with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed
- Appropriate NVIDIA drivers (compatible with CUDA 12.8)
- If you want to run the examples, set your Hugging Face token env var `HF_TOKEN` in your local startup (.bashrc, .zshrc or .profile file)

## Quick Start
1. Build the container image

```bash
./container/build.sh --target local-dev
```

The container will be built and give certain file permissions to your local uid and gid.

> Note: Currently local-dev is only implemented for --framework VLLM

2. Open Dynamo folder in VS Code
- Press Ctrl + Shift + P
- Select "Dev Containers: Open Folder in Container"

If you want to mount your hugging face cache, go to `.devcontainer` and uncomment in the mounts section:

```json
// "source=${localEnv:HF_HOME},target=/home/ubuntu/.cache/huggingface,type=bind", // Uncomment to enable HF Cache Mount. Make sure to set HF_HOME env var in you .bashrc
```
Make sure HF_HOME is sourced in your .bashrc or .zshenv and your vscode default terminal is set properly.

3. Wait for Initialization
- The container will mount your local code
- `post-create.sh` will build the project and configure the environment

If `post-create.sh` fails, you can try to debug or [submit](https://github.com/ai-dynamo/dynamo/issues) an issue on github.

## What's Inside
Development Environment:
- Rust and Python toolchains
- GPU acceleration
- VS Code extensions for Rust and Python
- Persistent build cache in `.build/` directory enables fast incremental builds (only changed files are recompiled)

`cargo build --locked --profile dev` to re-build

- Edits to files are propogated to local repo due to the volume mount
- SSH and GPG agent passthrough orchestrated by devcontainer

File Structure:
- Local dynamo repo mounts to `/home/ubuntu/dynamo`
- Python venv in `/opt/dynamo/venv`
- Build artifacts in `dynamo/.build/target`
- HuggingFace cache preserved between sessions (Mounting local path `HF_HOME` at `/home/ubuntu/.cache/huggingface`)
- Bash memory preserved between sessions at `/home/ubuntu/.commandhistory` using docker volume `dynamo-bashhistory`
- Precommit peeserved between sessions at `/home/ubuntu/.cache/precommit` using docker volume `dynamo-precommit-cache`

## Customization
Edit `.devcontainer/devcontainer.json` to modify:
- VS Code settings and extensions
- Environment variables
- Container configuration
- Custom Mounts

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

### Environment Variables Not Set in Container?

If your environment variables are not being set in your devcontainer (e.g., `echo $HF_TOKEN` returns empty), and these variables are defined in your `~/.bashrc`, there are two ways to ensure they are properly sourced:

1. Add `source ~/.bashrc` to your `~/.bash_profile`, OR
2. Add `source ~/.bashrc` to your `~/.profile` AND ensure `~/.bash_profile` does not exist

Note: If both `~/.bash_profile` and `~/.profile` exist, bash will only read `~/.bash_profile` for login shells. Therefore, if you choose option 2, you must remove or rename `~/.bash_profile` to ensure `~/.profile` (and consequently `~/.bashrc`) is sourced.


See VS Code Dev Containers [documentation](https://code.visualstudio.com/docs/devcontainers/containers) for more details.
