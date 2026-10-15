/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Workload is the immutable result of resolving and naming one LPX component group
// for scheduling and runtime materialization.
// Its methods require a non-nil workload returned by ResolveWorkloads.
type Workload struct {
	models               []*Model
	digest               WorkloadDigest
	scalingGroupReplicas int32
	conductorReplicas    int32
	groupName            string
	plan                 *MaterializationPlan
}

// Models returns a read-only view of the workload's model projections.
// Callers must not modify the slice or its projections.
func (w *Workload) Models() []*Model {
	return w.models
}

// Digest returns the workload's digest.
func (w *Workload) Digest() WorkloadDigest {
	return w.digest
}

// Pipeline returns the workload's runtime pipeline.
func (w *Workload) Pipeline() Pipeline {
	return w.models[0].pipeline
}

// ComponentNames returns this workload's members in canonical runtime order.
func (w *Workload) ComponentNames() []string {
	// Draft fanout can contribute several consecutive models from one component.
	var names []string
	for _, projection := range w.models {
		if len(names) == 0 || names[len(names)-1] != projection.componentName {
			names = append(names, projection.componentName)
		}
	}
	return names
}

// ServingComponentName returns the conductor owner, placed last during resolution.
func (w *Workload) ServingComponentName() string {
	return w.models[len(w.models)-1].componentName
}

// WithNames returns an immutable workload named inside pcsName. group is empty
// for a sole workload, or its serving component when the PCS contains several.
func (w *Workload) WithNames(pcsName, group string) (*Workload, error) {
	plan, err := w.PlanNodeLocalMaterialization(pcsName)
	if err != nil {
		return nil, err
	}
	if group != "" {
		plan, err = plan.WithGroup(group)
		if err != nil {
			return nil, err
		}
	}
	named := *w
	named.plan = plan
	named.groupName = group
	if err := named.ValidateReplicas(plan.Replicas); err != nil {
		return nil, err
	}
	return &named, nil
}

// GroupName returns the serving component used to scope a multi-workload identity,
// or an empty string for a sole workload.
func (w *Workload) GroupName() string { return w.groupName }

// ScalingGroup returns this named workload's materialized Grove group.
func (w *Workload) ScalingGroup() string { return w.plan.LPXScalingGroup }

// ResourcePrefix scopes runtime resources to this named workload.
func (w *Workload) ResourcePrefix() string { return w.plan.ResourcePrefix }

// ConductorTemplate returns the named Nova or Cyborg clique template.
func (w *Workload) ConductorTemplate() string {
	if w.Pipeline() == PipelineHybrid {
		return w.plan.CyborgTemplate
	}
	return w.plan.ConductorTemplate
}

// Replicas returns the authored scaling-group capacity or its initial seed.
func (w *Workload) Replicas() int32 { return w.scalingGroupReplicas }

// ConductorReplicas returns the resolved Nova or Cyborg clique width.
func (w *Workload) ConductorReplicas() int32 { return w.conductorReplicas }

// ConductorCliqueName returns the serving clique in one scaling-group replica.
func (w *Workload) ConductorCliqueName(replica int32) string {
	return materializedCliqueNameForReplica(w.ScalingGroup(), w.ConductorTemplate(), replica)
}

// ValidateReplicas validates group capacity and complete Cyborg client groups.
func (w *Workload) ValidateReplicas(replicas int32) error {
	plan := *w.plan
	plan.Replicas = replicas
	if err := plan.ValidateReplicaCount(); err != nil {
		return err
	}
	if w.Pipeline() == PipelineHybrid {
		return w.ValidateCyborgReplicas(w.conductorReplicas)
	}
	return nil
}

// RequestSpec returns a fresh request for a model index and workload replica.
// index must be in Models; replica must be within the validated live capacity.
func (w *Workload) RequestSpec(index int, replica int32) lpxv1alpha1.LPUPipelineRequestSpec {
	plan := w.plan.ForReplica(replica)
	return w.models[index].RequestSpec(plan, plan.Agents[index].CliqueName)
}

// Render appends the workload to pcs only after its complete rendering succeeds.
// The workload and PCS are resolved and nonnil; input contains fresh owned templates.
func (w *Workload) Render(pcs *grovev1alpha1.PodCliqueSet, input RenderInput) ([]client.Object, error) {
	if w.plan == nil {
		return nil, fmt.Errorf("workload resource names must be resolved before rendering")
	}
	rendered, err := RenderNodeLocal(w, w.plan, input)
	if err != nil {
		return nil, err
	}
	pcs.Spec.Template.Cliques = append(pcs.Spec.Template.Cliques, rendered.Cliques...)
	pcs.Spec.Template.PodCliqueScalingGroupConfigs = append(pcs.Spec.Template.PodCliqueScalingGroupConfigs, rendered.ScalingGroup)
	return rendered.Resources, nil
}
