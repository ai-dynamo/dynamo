//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// Projection tests supply historical inputs explicitly; production loads them from the checkpoint.
func reconcileEngineGroupDesiredAssignment(group *api.DynamoGraphDeploymentEngineGroup, profile api.EngineGroupProfileStatus, state domain.GroupStatus, plan *domain.ResolvedPlan) {
	c := engineGroupCheckpoint{State: state, LastStableReplicas: group.Status.LastStableReplicas,
		LastStableTopologyGeneration: group.Status.LastStableTopologyGeneration}
	c.DesiredNativeMembers, c.DesiredAssignmentGeneration = resolveEngineGroupDesiredAssignment(
		group, profile, state, plan, group.Status.DesiredNativeMembers, group.Status.DesiredAssignmentGeneration)
	if c.isStable(group, profile) {
		c.LastStableReplicas = int32(len(state.Capacity.Observed.Allocations))
		c.LastStableTopologyGeneration = state.Membership.Observed.CommittedTopology.Generation
	}
	c.projectHistory(group)
}

func projectEngineGroupOperation(group *api.DynamoGraphDeploymentEngineGroup, state domain.GroupStatus) error {
	operation, err := resolveEngineGroupOperation(group, state, group.Status.Operation)
	if err != nil {
		return err
	}
	group.Status.Operation = operation
	return nil
}
