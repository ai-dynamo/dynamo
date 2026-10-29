<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md - global-discovery-mcs end-to-end test

The kind end-to-end test of the registry, the controller and (phase 3) the
Global Discovery Service. Read `README.md` for the setup, the phases and the
settings. It runs through `make -C deploy/global-discovery-mcs e2e` and
`e2e-down`; the Makefile installs `kind` into `bin/`.

## Commands

From `deploy/global-discovery-mcs`:

```bash
make e2e                           # all phases; builds the images, creates three clusters, deletes them at the end
make e2e E2E_PHASES=2 E2E_KEEP=always E2E_SKIP_BUILD=1   # iterate on one phase against kept clusters
make e2e-down                      # delete the clusters
shellcheck -x test/e2e/*.sh        # must stay clean
```

Run it before changing the controller, the chart values it uses, or the way the
Global Discovery Service reads the imports.

## Rules

- Bash 3.2 compatible (macOS): no associative arrays, `mapfile` or `${x,,}`.
  `set -euo pipefail` in `run.sh`; the other files are sourced by it. Every
  file has the dated two-line SPDX header (after the shebang for scripts);
  Markdown stays ASCII; shellcheck must be clean.
- A new assertion is a check function that returns 0 when satisfied and sets
  `LAST_OBSERVED`, run through `expect` (polls up to `E2E_TIMEOUT`) or `once`
  (steps that write). No fixed sleeps.
- Never touch anything but the clusters `<E2E_PREFIX>-hub|a|b`. Every kubectl and
  helm call goes through `kc KEY` or `hm KEY`, which pass the kubeconfig file in
  `bin/e2e` and the kind context, never the default kubeconfig. Never change
  global machine settings from the script (the inotify limit is only checked).
- Each phase resets its own namespaces (`reset_namespace`) and does not depend
  on another phase. Steps that need cluster b are skipped without it
  (`has_cluster b`).
- Registry and Global Discovery Service calls go through the API server's
  Service proxy of the hub, not a port-forward.
- Images are tagged `e2e-<image id>` so a rebuilt image always reaches a reused
  cluster. Pin tool and node image versions in variables; `KIND_VERSION` lives in
  the module Makefile.
- A documented behaviour the Global Discovery Service lacks is a `gap` (a GAP
  line, a failure under `E2E_STRICT=1`), not a loosened assertion. Remove the
  gap when the service gets the behaviour. Do not change the module or the Rust
  service to make the test pass.
- Label, annotation, condition and name values come from `api/labels.go` and
  `api/errors.go`; the test mirrors them and fails when they drift.
