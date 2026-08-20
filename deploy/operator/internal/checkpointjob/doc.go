// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package checkpointjob builds the batch/v1 Job that runs a DynamoCheckpoint's
// target pod under CRIU/cuda-checkpoint capture, and shapes restore pods once
// a checkpoint is ready.
//
// Ported from github.com/ai-dynamo/snapshot's operator/internal/protocol
// package, which is Go-internal and so not importable from Dynamo. It is now
// Dynamo-owned: there is no upstream source of truth for it, and fixes made
// in the Snapshot repo's own copy do not flow here automatically.
// github.com/ai-dynamo/snapshot's SnapshotJob CRD (tracked there as
// RUN-39806) is the planned long-term replacement for this package once it
// ships.
//
// (trivial touch to trigger DynamoCheckpoint CI on this branch)
package checkpointjob
