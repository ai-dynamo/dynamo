<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Sharing the source

Keep deployment settings in `*.local.json` files. Local credentials, CDK context,
synthesized templates, build outputs, downloaded build assets, test results and
the entire `cdk/testing/experiments/` tree are ignored. Experiment scripts and
reports can contain live endpoints and resource IDs even when they contain no
credentials. Keep reusable test code in the other `cdk/testing/` directories.

Create a source-only directory from `examples/global-router/`:

```bash
python3 cdk/scripts/export_public.py /tmp/global-router-public
```

The destination must not exist and must be outside this repository. The exporter
copies only this example’s existing Git-tracked and nonignored untracked files, including current
edits; it omits Git history and refuses tracked files covered by ignore rules or
symlinks. It does not scan or redact file contents. Review and secret-scan the
export before sharing, especially after adding new configuration or examples.
Do not upload a ZIP of the working directory: Git ignore rules do not apply to
ordinary directory copies.

For example, with Gitleaks installed:

```bash
gitleaks dir /tmp/global-router-public --redact --max-archive-depth 3
gitleaks git /path/to/dynamo --log-opts=--all --redact
```

The history scan matters if you intend to publish the existing Git repository.
Ignoring or deleting a file does not remove previously committed copies. Review
the staged file list and diff before committing. A clean scanner result does not
detect every possible secret or identify all private operational information.
