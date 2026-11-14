/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package workload defines physical allocation independently of engine membership and workload providers.
package workload

import (
	"context"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// Allocation identifies physical capacity without assigning native members or process incarnations.
// The supported profiles allocate one Pod per slot; a replacement has a different Pod UID.
// Index-based bootstrap profiles use canonical slot-N identities; providers translate their
// native indices to that contract instead of exposing native Pod names or labels to engines.
type Allocation struct {
	ReplicaID enginegroup.ReplicaID
	SlotID    enginegroup.CapacitySlotID
	Pod       enginegroup.CapacityRef
	Ready     bool
	Health    enginegroup.AllocationHealth
	// TemplateRevision is an opaque immutable launch revision used in engine-profile fingerprints.
	TemplateRevision string
}

// Observation is fresh physical evidence, not engine membership or inference readiness.
type Observation struct {
	AppliedRevision int64
	Allocations     []Allocation
	ReleaseFences   []enginegroup.ReleaseFence
	ReleaseMode     enginegroup.CapacityReleaseMode
}

// Capacity owns physical allocation and release. Native-member and bootstrap semantics are opaque here;
// engine integrations validate those before delegating effects. Prepare reconciles provider-owned
// allocation prerequisites without starting a membership operation or granting serving permission.
type Capacity interface {
	Prepare(ctx context.Context) error
	Observe(ctx context.Context, groupID enginegroup.GroupID) (Observation, error)
	Apply(ctx context.Context, groupID enginegroup.GroupID, target enginegroup.CapacityTarget) (enginegroup.ApplyResult, error)
}
