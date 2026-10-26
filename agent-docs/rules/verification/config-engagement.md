<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Config Engagement

Before running AIPerf, prove that the intended change reached the live serving process. Confirm the DGD reconciled,
pods rolled when required, and the expected replicas, resources, arguments, environment, or runtime behavior are active.

Applied YAML and a passing smoke request are not sufficient by themselves. Record durable Kubernetes status, pod spec,
or startup-log evidence. Do not benchmark or promote an unconfirmed change.
