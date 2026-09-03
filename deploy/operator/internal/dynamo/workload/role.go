/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package workload contains shared rendering vocabulary for materialized
// Kubernetes workloads.
package workload

// Role identifies the workload role currently being rendered.
type Role string

const (
	RoleLeader     Role = "leader"
	RoleWorker     Role = "worker"
	RoleMain       Role = "main"
	RoleCheckpoint Role = "checkpoint"
	RoleGMS        Role = "gms"
)
