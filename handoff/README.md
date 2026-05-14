# Dynamo production Kubernetes handoff

Self-contained handoff kit for resuming work on `ai-blaise/dynamo-prod-k8s`.
It mirrors the structure of `ion-w5-handoff`, but this repo does not need a
checked-in binary state bundle: GitHub is the source of truth, and deployment
state lives in Kubernetes, Google Cloud, Hugging Face, and the production
manifests under `deploy/production`.

## What's in here

| File | Purpose |
|---|---|
| `MANIFEST.md` | Current repo context, protected custom surfaces, verification commands, and deployment rules |
| `unpack-state.sh` | Bootstrap script for a fresh workstation or agent environment |
| `pack-state.sh` | Creates a small timestamped handoff bundle with repo metadata and selected docs |
| `setup/setup-all.sh` | Runs the setup scripts in order |
| `setup/setup-prereqs.sh` | Checks or installs common repo tools |
| `setup/setup-github.sh` | Installs and authenticates GitHub CLI, then verifies git identity |
| `setup/setup-gcloud.sh` | Installs and authenticates Google Cloud SDK, then sets the default project |
| `setup/setup-huggingface.sh` | Installs Hugging Face CLI support and verifies authentication |

## One-shot setup on a fresh machine

```bash
git clone https://github.com/ai-blaise/dynamo-prod-k8s.git ~/dynamo-prod-k8s
cd ~/dynamo-prod-k8s

# Interactive: GitHub, Google Cloud, and Hugging Face auth may require browser
# flows or token paste.
bash handoff/setup/setup-all.sh

# Clone/fetch the repo into a handoff workspace and print the project context.
bash handoff/unpack-state.sh
```

The setup scripts are idempotent. They are safe to rerun after fixing a missing
tool or completing an auth flow.

## Agent prompt

```text
Clone https://github.com/ai-blaise/dynamo-prod-k8s and read:
  AGENTS.md
  handoff/README.md
  handoff/MANIFEST.md
  deploy/production/README.md
  deploy/production/docs/smg-integration.md
  docs/components/frontend/Tokenizer.md
  docs/features/tokenizer/README.md

If GitHub, Google Cloud, or Hugging Face auth is missing, ask the user to run:
  bash handoff/setup/setup-all.sh

Then run:
  bash handoff/unpack-state.sh

Before changing code:
1. Explicitly check https://github.com/ai-blaise/dynamo-prod-k8s for the live
   fork state.
2. Check https://github.com/ai-dynamo/dynamo for upstream main.
3. Preserve all fork-owned production integrations under deploy/production.
4. Preserve the fork contract that fastokens is the default tokenizer and
   --tokenizer default is the HuggingFace opt-out.
5. Use instance-20260415-161450 in asia-south1-b for deployment and verification.
```

## What `unpack-state.sh` does

1. Verifies `git`, `curl`, `python3`, and optional auth tools.
2. Clones or updates `ai-blaise/dynamo-prod-k8s`.
3. Adds the read-only upstream remote `ai-dynamo/dynamo`.
4. Prints origin and upstream commit state.
5. Verifies the protected production directory and FastOkenS files exist.
6. Prints the standard verification commands and VM target.

It does not copy credentials, tokens, model caches, kubeconfigs, or cloud state.

## Re-snapshotting

When a handoff needs a frozen metadata snapshot:

```bash
cd ~/dynamo-prod-k8s
bash handoff/pack-state.sh
```

The generated `dynamo-prod-k8s-state-bundle-*.tar.gz` contains a manifest,
current git metadata, selected production docs, and file inventories. It is
intended for transfer between workstations, not as a replacement for GitHub.

## Why this exists

This fork carries production Kubernetes integrations that upstream Dynamo does
not own. The handoff kit makes those fork-specific contracts explicit so future
agents can merge upstream Dynamo changes without deleting or weakening the
custom production stack.
