<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Slime External Rollouts

**Experimental.** This directory runs Slime against a fixed set of SGLang engines with Dynamo
sidecars, managed by a `DynamoGraphDeployment`.

For prerequisites, deployment, engine readiness, and launch steps, see
[Slime](../../../docs/fern/pages/use-cases/reinforcement-learning/slime.md).

## Contents

| Path | Description |
|---|---|
| `dynamo.yaml` | DGD template for the fixed SGLang engine set with Dynamo sidecars |
| `deploy-dynamo.sh` | Deploys the fixed Dynamo SGLang worker set |
| `launch-slime.sh` | Starts Slime against the fixed SGLang engines |
| `dynamo_generate.py` | Custom Slime generator that streams rollout requests through the Dynamo frontend |
