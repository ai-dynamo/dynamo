<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Operator Helm Chart

This standalone chart installs the Dynamo Kubernetes operator. The Dynamo platform chart also
includes it as a dependency.

## External Grove prerequisite

The standalone chart does not install Grove or run the platform chart's Grove compatibility
preflight. Before installing or upgrading the operator with Grove integration enabled, ensure your
deployment gate verifies both of the following:

- The Grove controller satisfies the version required by the matching Dynamo platform chart.
- The matching Grove CRDs are installed, and the controller Deployment completed its rollout.

Upgrade the Grove controller and its CRDs together before applying the Dynamo operator chart.
