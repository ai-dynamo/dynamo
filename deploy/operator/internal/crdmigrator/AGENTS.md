<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CRD Migrator Maintenance

- This package is derived from
  `kubernetes-sigs/cluster-api/controllers/crdmigrator` at Cluster API v1.13.4,
  commit `27f464418c195d96ae2ef4b96f3b6a047ea89310`.
- The TTL cache is derived from `kubernetes-sigs/cluster-api/util/cache/cache.go`
  at the same version and commit. Keep its expiration and periodic sweep
  behavior aligned with upstream.
- Keep the package free of `sigs.k8s.io/cluster-api` imports. It may depend on
  the Go standard library, Kubernetes libraries, and controller-runtime.
- Preserve the upstream and NVIDIA copyright notices and the source reference
  in derived files.
- When controller-runtime is upgraded, compare this package and its tests with
  the current Cluster API migrator. Pay particular attention to controller
  setup, server-side apply, pagination, conflict handling, managedFields
  cleanup, and storage-version completion semantics.
- Port relevant upstream bug fixes and tests. Document intentional differences
  in the source and keep them as small as possible.
