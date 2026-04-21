<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Claude Code & agent guide for ai-dynamo/dynamo

This file orients Claude Code, Cursor, and other agentic assistants working in
this repo. Humans should start at [`README.md`](README.md).

## What this repo is

Dynamo is the open-source, datacenter-scale inference orchestration layer above
SGLang, TensorRT-LLM, and vLLM. Most readers of this repo are *deploying* and
*operating* Dynamo. A smaller (but important) audience contributes back. Route
your help based on which one you're talking to.

## If you're helping a Dynamo *user*

| Goal | Start here |
|---|---|
| Deploy a recipe end-to-end | [`.claude/skills/quickstart/SKILL.md`](.claude/skills/quickstart/SKILL.md) (or [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md) for the human-facing guide) |
| Preflight the cluster before deploying | [`.claude/skills/verify-cluster/SKILL.md`](.claude/skills/verify-cluster/SKILL.md) |
| Inspect a running deployment | [`.claude/skills/inspect-pods/SKILL.md`](.claude/skills/inspect-pods/SKILL.md) |
| Diagnose a failure | [`.claude/skills/troubleshoot/SKILL.md`](.claude/skills/troubleshoot/SKILL.md) (skill) or [docs/troubleshooting.md](docs/troubleshooting.md) (human-facing) |
| Pick or compose an example | [examples/README.md](examples/README.md) (use-case index) |
| Apply a production recipe | [recipes/README.md](recipes/README.md) |
| Pick the right backend | [docs/backends/](docs/backends/) |
| Build a custom image, swap storage class, enable observability | [SKILLS.md](SKILLS.md) "Operations cookbook" (covered inline by `quickstart`) |

## If you're helping a Dynamo *contributor*

| Goal | Start here |
|---|---|
| Repo conventions, DCO, sign-off | [CONTRIBUTING.md](CONTRIBUTING.md) |
| File a bug, propose a DEP, monitor a PR | [SKILLS.md](SKILLS.md) "Contributor skills" |
| Update the docs site | [.claude/skills/dynamo-docs/SKILL.md](.claude/skills/dynamo-docs/SKILL.md) |
| Run / extend the Kubernetes operator | [deploy/cloud/operator/](deploy/cloud/operator/) |

## Conventions for agents

- Container images under `nvcr.io/nvidia/ai-dynamo/*` (vllm-runtime,
  sglang-runtime, tensorrtllm-runtime, mocker-runtime) are public on NGC.
  Do not invent `imagePullSecrets` steps unless the user is pulling from a
  private registry mirror or a non-public third-party image.
- Treat `.claude/skills/<name>/SKILL.md` as the authoritative procedure for any
  skill listed in `SKILLS.md`. The table in `SKILLS.md` is an index, not the
  procedure.
- Use `examples/` for "show me how this works" code; use `recipes/` for
  "deploy this for production." Don't conflate the two.
- Backend feature support lives in [docs/backends/](docs/backends/) and the
  [Feature Support Matrix](docs/backends/trtllm/README.md#feature-support-matrix),
  not in individual launch scripts.
- User-facing skill modules live alongside the contributor ones under
  `.claude/skills/` (e.g. `quickstart`, `verify-cluster`, `inspect-pods`,
  `troubleshoot`). The inline procedures in `SKILLS.md` are the same content
  in narrative form for users without an agent CLI.

## Where not to look

- `.devin/`, `.cursor/`, `.claude/settings.local.json`: per-developer state,
  gitignored.
- `worklogs/`: per-session debug worklogs (see the `debug-session` skill),
  gitignored.
