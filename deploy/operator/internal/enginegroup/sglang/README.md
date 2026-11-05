<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native SGLang Engine Group adapters

This package implements SGLang membership control, native traffic observation,
serving verification and growth planning against the Engine Group contracts.
It does not render workloads or depend on a workload provider.

The native API requires runtime, operation, joining-allocation and base-topology
correlation from [SGLang #39834](https://github.com/sgl-project/sglang/pull/39834).
Growth observes before replaying the identical request. Pending or ambiguous
submission is not a definitive rejection, and current runtime health is checked
independently of retained operation results.

The caller supplies allocation evidence and immutable geometry. Private journals
retain Dynamo control correlation; the engine remains authoritative for native
membership and operation outcomes. Traffic observation represents engine-internal
admission, not independent discovery-registration evidence. Serving verification
does not grant registration.

CPU tests exercise these contracts with native HTTP fixtures and allocation
observations. They do not establish GPU serving, shrink or failure recovery.

```bash
go test -race ./internal/enginegroup/sglang/...
```
