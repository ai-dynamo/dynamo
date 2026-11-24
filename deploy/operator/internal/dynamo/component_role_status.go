// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"encoding/json"
	"maps"
	"slices"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

// Bound this observational export independently of the replay allocation budget.
const maxComponentRoleStatusBytes = 128 << 10

// ResolveComponentRoleStatus projects one already lowered PodSpec without
// interpreting its command or reading runtime inputs. podSpec must not be nil.
func ResolveComponentRoleStatus(name string, podSpec *corev1.PodSpec) v1beta1.ComponentRoleStatus {
	remaining := maxComponentRoleStatusBytes
	return resolveComponentRoleStatus(name, podSpec, &remaining)
}

func resolveComponentRoleStatus(name string, podSpec *corev1.PodSpec, remaining *int) v1beta1.ComponentRoleStatus {
	return v1beta1.ComponentRoleStatus{Name: name, PodSpec: corev1.PodSpec{
		Containers:     resolveTargetContainers(podSpec.Containers, remaining),
		InitContainers: resolveTargetContainers(podSpec.InitContainers, remaining),
	}}
}

func resolveTargetContainers(containers []corev1.Container, remaining *int) []corev1.Container {
	if len(containers) == 0 {
		return nil
	}

	// Both native PodSpec lists share the retained expansion budget.
	projected := make([]corev1.Container, 0, len(containers))
	for _, container := range containers {
		// Replay the actual final list, never the authored layers or a sorted map.
		status := corev1.Container{
			Name: container.Name, Image: container.Image,
			Command: slices.Clone(container.Command), Args: slices.Clone(container.Args),
		}
		values := semanticenv.Replay(container.Env)
		last := make(map[string]int, len(values))
		for i, variable := range container.Env {
			last[variable.Name] = i
		}

		// Preserve each final native binding and its source when evaluation is unavailable.
		for i, variable := range container.Env {
			if last[variable.Name] != i {
				continue
			}
			binding := *variable.DeepCopy()
			value := values[variable.Name]

			// JSON escaping costs at most six bytes per input byte. Bound retained
			// expansions across containers before allocating a serialization buffer.
			if value.Known && len(value.Value) <= *remaining/6 {
				binding.Value = value.Value
				*remaining -= 6 * len(value.Value)
			}
			status.Env = append(status.Env, binding)
		}
		projected = append(projected, status)
	}
	return projected
}

// ResolveGroveComponentRoleStatuses projects final target templates using the
// same role expansion as rendering. Ambiguous per-rank shapes remain unreported.
// dgd and pcs must not be nil; delegated components have no matching cliques.
func ResolveGroveComponentRoleStatuses(dgd *v1beta1.DynamoGraphDeployment, pcs *grovev1alpha1.PodCliqueSet) map[string][]v1beta1.ComponentRoleStatus {
	statuses := make(map[string][]v1beta1.ComponentRoleStatus)
	remaining := maxComponentRoleStatusBytes
	for i := range dgd.Spec.Components {
		before := remaining
		component := &dgd.Spec.Components[i]
		roles := expandRolesForComponent(component.ComponentName, component.Replicas, component.GetNumberOfNodes(), component)
		seen := make(map[Role]bool, len(roles))
		var projected []v1beta1.ComponentRoleStatus
		for _, role := range roles {
			clique := cliqueByName(pcs, strings.ToLower(role.Name))
			if clique == nil || seen[role.Role] {
				projected = nil
				break
			}
			seen[role.Role] = true
			projected = append(projected, resolveComponentRoleStatus(string(role.Role), &clique.Spec.PodSpec, &remaining))
		}
		if projected == nil {
			remaining = before
		}
		statuses[component.ComponentName] = projected
	}
	return statuses
}

// BoundComponentRoleStatuses limits the combined serialized role export by
// clearing whole component projections in name order after the budget is used.
// It mutates only the caller-owned status map; a nil map is an empty projection.
// Whole-object headroom is checked separately at status publication.
func BoundComponentRoleStatuses(statuses map[string]v1beta1.ComponentReplicaStatus) {
	remaining := maxComponentRoleStatusBytes
	for _, name := range slices.Sorted(maps.Keys(statuses)) {
		status := statuses[name]
		data, err := json.Marshal(status.Roles)
		if err != nil || len(data) > remaining {
			status.Roles = nil
			statuses[name] = status
			continue
		}
		remaining -= len(data)
	}
}
