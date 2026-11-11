<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-router-protocol

`dynamo-router-protocol` defines the identities and wire types that cross a process boundary between the router and engines, workers, or sidecars: worker and routing-partition identities, session context, attempt ids, and the worker-type and column vocabulary. It contains data only (`serde` derives and conversions) and depends on nothing else in Dynamo, so every router crate and host can name it.
