/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
)

func (c *Coordinator) reconcileInitialAdmission(ctx context.Context, groupID GroupID, status GroupStatus, topology MembershipTopology) (bool, ReconcileResult, error) {
	// Registration follows the serving runtime's native evidence; no initial permission is synthesized.
	if sameMemberships(status.Traffic.Observed.Admitted, topology.Replicas) {
		return false, ReconcileResult{Status: status}, nil
	}
	if !sameTopologyWithCurrent(status.Topologies, topology) {
		return true, ReconcileResult{Status: status}, fmt.Errorf("initial topology changed without membership authority")
	}
	if err := validatePinnedCapacity(status.Registry, status.Capacity.Observed, topology); err != nil {
		return true, ReconcileResult{Status: status}, err
	}

	// Verification cannot bypass physical availability, even when its startup probe observation lags.
	for _, membership := range topology.Replicas {
		allocation, found := allocationByID(status.Capacity.Observed, membership.ReplicaID)
		if !found || !allocation.Available {
			return true, ReconcileResult{Status: status, Requeue: true}, nil
		}
	}
	if status.InitialServingProof == nil {
		result, err := c.verifier.Verify(ctx, groupID, topology)
		if err != nil {
			return true, ReconcileResult{Status: status, Requeue: true}, err
		}
		if result.Failure != nil || result.Proof == nil || result.Proof.TopologyGeneration != topology.Generation || result.Proof.RuntimeDigest != TopologyRuntimeDigest(topology) {
			return true, ReconcileResult{Status: status, Requeue: true}, fmt.Errorf("initial serving verification did not prove the exact topology")
		}
		status.InitialServingProof = cloneServingProof(result.Proof)
		return true, ReconcileResult{Status: status, Requeue: true}, nil
	}
	if status.InitialServingProof.TopologyGeneration != topology.Generation || status.InitialServingProof.RuntimeDigest != TopologyRuntimeDigest(topology) {
		return true, ReconcileResult{Status: status}, fmt.Errorf("initial serving proof no longer matches membership")
	}
	return true, ReconcileResult{Status: status, Requeue: true}, nil
}
