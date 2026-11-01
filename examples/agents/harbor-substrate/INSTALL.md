<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Platform installation is documented in
[Install Substrate](../../../docs/fern/pages/kubernetes/agent-sandboxes/install-substrate.mdx)
under **Kubernetes Guide → Agent Sandboxes**.

For this tutorial, run the installation commands from this directory and choose
**Use the Dynamo example** in the configuration step. Edit
[`substrate-install.yaml`](substrate-install.yaml) in place. This is input to
`ate-setup --config`, not a Kubernetes manifest for `kubectl apply`.
The installation page also supports an independent configuration for other workloads.

After installation, follow
[Run SWE-bench Verified with Harbor, Pi, and Substrate](../../../docs/fern/pages/use-cases/agents/harbor-substrate.mdx)
for Harbor access, guest images, task timeouts, and workload verification.
