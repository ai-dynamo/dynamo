// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package lpx

import (
	"context"
	"maps"
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// resolveWorkloads adapts resolved workloads to the existing scenario fixture DSL.
// The production resolver owns validation and naming.
func (r *graphReconciler) resolveWorkloads(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, dgd *v1beta1.DynamoGraphDeployment) (map[string]*lpx.Workload, map[string]*lpx.MaterializationPlan, error) {
	resolved, err := lpx.ResolveWorkloads(ctx, r.Client, dgd, dynamo.PCSNameForLPX(deployment), r.modelRegistry)
	if err != nil {
		return nil, nil, err
	}
	workloads := make(map[string]*lpx.Workload, len(resolved))
	plans := make(map[string]*lpx.MaterializationPlan, len(resolved))
	for _, workload := range resolved {
		name := workload.ServingComponentName()
		plan, err := workload.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(deployment))
		if err != nil {
			return nil, nil, err
		}
		if len(resolved) > 1 {
			plan, err = plan.WithGroup(name)
			if err != nil {
				return nil, nil, err
			}
		}
		workloads[name], plans[name] = workload, plan
	}
	return workloads, plans, nil
}

func (r *graphReconciler) renderPodCliqueSet(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, dgd *v1beta1.DynamoGraphDeployment, workloads map[string]*lpx.Workload, plans map[string]*lpx.MaterializationPlan) (*grovev1alpha1.PodCliqueSet, []client.Object, error) {
	named := make([]*lpx.Workload, 0, len(workloads))
	for _, name := range slices.Sorted(maps.Keys(workloads)) {
		group := ""
		if len(workloads) > 1 {
			group = name
		}
		workload, err := workloads[name].WithNames(plans[name].PodCliqueSetName, group)
		if err != nil {
			return nil, nil, err
		}
		named = append(named, workload)
	}
	return r.renderWorkloads(ctx, deployment, dgd, named, nil)
}
func resolvePipelineRequests(deployment *v1alpha1.LPXGraphDeployment, current map[string]*lpxv1alpha1.LPUPipelineRequest, workload *lpx.Workload, plan *lpx.MaterializationPlan) (map[string]*lpxv1alpha1.LPUPipelineRequest, []*lpxv1alpha1.LPUPipelineRequest) {
	group := ""
	if plan.ResourcePrefix != plan.PodCliqueSetName {
		group = workload.ServingComponentName()
	}
	named, err := workload.WithNames(plan.PodCliqueSetName, group)
	if err != nil {
		panic(err)
	}
	return resolveNamedPipelineRequests(deployment, current, named, plan.Replicas)
}
