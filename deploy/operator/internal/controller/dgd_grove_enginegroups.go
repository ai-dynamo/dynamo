/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
)

func (r *groveWorkloadsReconciler) reconcileEngineGroups(ctx context.Context, dgd *api.DynamoGraphDeployment, pcs *grove.PodCliqueSet, result *ReconcileResult) error {
	// The DGD seeds worlds once; the Engine Group controller alone owns subsequent member-clique targets.
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.EngineGroup == nil {
			continue
		}
		desired, err := dynamo.RenderComponentEngineGroup(dgd, component, 0)
		if err != nil {
			return fmt.Errorf("component %q: %w", component.ComponentName, err)
		}
		groupStatus, ready, err := r.engineGroups.Reconcile(ctx, dgd, pcs, desired)
		if err != nil {
			return err
		}

		// Pod readiness alone never makes the parent world Ready or resets its live target.
		if result.ComponentStatus == nil {
			result.ComponentStatus = make(map[string]api.ComponentReplicaStatus)
		}
		status := result.ComponentStatus[component.ComponentName]
		status.Replicas = groupStatus.Replicas
		status.ReadyReplicas = groupStatus.ReadyReplicas
		status.AvailableReplicas = groupStatus.ReadyReplicas
		status.GPUsPerReplica = groupStatus.GPUsPerReplica
		status.GPUsPerEngine = groupStatus.GPUsPerEngine
		result.ComponentStatus[component.ComponentName] = status
		if !ready {
			result.State = api.DGDStatePending
			result.Reason = "engine_groups_not_ready"
			result.Message = "Waiting for Engine Group binding and verified serving membership"
		}
	}
	return nil
}
