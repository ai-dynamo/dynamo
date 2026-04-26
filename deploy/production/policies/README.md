<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Production Network Policies

The `kubernetes/` directory contains portable `NetworkPolicy` examples for the `dynamo-system` namespace. They are intentionally not included in the default Argo CD root app because API server, registry, object-store, model-store, NATS, and service-mesh egress needs vary by cluster.

The `calico/` directory contains a Calico `GlobalNetworkPolicy` variant for clusters that use Calico policy tiers and ordered rules.

Apply these after confirming the cluster-specific egress destinations for:

- Kubernetes API server
- container registry
- object storage
- Hugging Face or internal model store
- external NATS, etcd, or Model Express endpoints
- service mesh control plane, when enabled
