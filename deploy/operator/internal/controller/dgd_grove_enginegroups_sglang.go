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

func (r *groveWorkloadsReconciler) reconcileEngineGroups(
	ctx context.Context,
	dgd *api.DynamoGraphDeployment,
	pcs *grove.PodCliqueSet,
) (map[string]api.ComponentReplicaStatus, bool, error) {
	statuses := make(map[string]api.ComponentReplicaStatus)
	ready := true
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.EngineGroup == nil {
			continue
		}

		// Resolve the supported backend before the shared lifecycle creates or binds capacity.
		profile, err := dynamo.ResolveComponentEngineGroupProfile(component)
		if err != nil {
			return nil, false, fmt.Errorf("component %q: %w", component.ComponentName, err)
		}
		desired := dynamo.RenderSGLangEngineGroup(dgd, component, 0, profile)
		status, groupReady, err := r.engineGroups.Reconcile(ctx, dgd, pcs, desired)
		if err != nil {
			return nil, false, err
		}
		statuses[component.ComponentName] = status
		ready = ready && groupReady
	}
	return statuses, ready, nil
}
