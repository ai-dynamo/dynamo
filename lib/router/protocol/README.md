<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-router-protocol

`dynamo-router-protocol` defines the types every router crate shares: the wire types that cross a process boundary between the router and engines, workers, or sidecars (worker and routing-partition identities, the worker-type and column vocabulary), and in-process identities such as session context and attempt ids. It contains data only (`serde` derives and conversions) and depends on nothing else in Dynamo, so every router crate and host can name it.
