# NVIDIA Dynamo Development Environment
## Prerequisites
- Docker installed and configured on your host system
- Visual Studio Code with the Dev Containers extension installed
- Appropriate NVIDIA drivers (compatible with CUDA 12.8)
- `HF_HOME`, `GITHUB_TOKEN`, and `HF_TOKEN` environment variables

## Quick Start
1. Build the container image

```bash
./container/build.sh --target local-dev
```

> Note: Currently local-dev is only implemented for --framework VLLM

2. Open Dynamo folder in VS Code
- Press Ctrl + Shift + P
- Select "Dev Containers: Open Folder in Container"

3. Wait for Initialization
- The container will mount your local code
- `post-create.sh` will build the project and configure the environment

## What's Inside
Development Environment:
- Rust and Python toolchains
- GPU acceleration
- VS Code extensions for Rust and Python
- Persistent build cache in `.build/` directory enables fast incremental builds (only changed files are recompiled)

`cargo build --profile dev --locked` to re-build

- Edits to files are propogated to local repo due to the volume mount
- Mounted .gitconfig to propogate user information for git commits
- GPG keys passthrough for signed commits from the docker container

File Structure:
- Local dynamo repo mounts to `/home/ubuntu/dynamo`
- Python venv in `/opt/dynamo/venv`
- Build artifacts in .build/target
- HuggingFace cache preserved between sessions (Mounted to `HF_HOME`)
- Bash memory preserved between sessions

## Customization
Edit `.devcontainer/devcontainer.json` to modify:
- VS Code settings and extensions
- Environment variables
- Container configuration

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

### SSH Keys for Git Operations

SSH keys need to be loaded in your SSH agent to work properly in the container:

```bash
# In devcontainer, Check if your keys are loaded in the agent
ssh-add -l

# On local host, if your key isn't listed, add it
eval "$(ssh-agent)"  # Start the agent if not running
ssh-add ~/.ssh/id_rsa

# On local host, for automatic loading at login (Option 1 - with keychain)
sudo apt install keychain  # Or brew install keychain on macOS
echo 'eval "$(keychain --eval --quiet --agents ssh id_rsa)"' >> ~/.bashrc

# On local host, for automatic loading at login (Option 2 - without keychain)
echo 'eval "$(ssh-agent)" > /dev/null' >> ~/.bashrc
echo 'ssh-add ~/.ssh/id_rsa > /dev/null' >> ~/.bashrc
```

Verify access by running `ssh -T git@github.com` in both host and container.


See VS Code Dev Containers [documentation](https://code.visualstudio.com/docs/devcontainers/containers) for more details.