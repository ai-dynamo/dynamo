// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"maps"
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// LivePowerCapSupport reports whether the graph supports runtime power caps. dgd must be non-nil.
func LivePowerCapSupport(dgd *v1beta1.DynamoGraphDeployment) (reason string, supported bool) {
	provider := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]
	if provider != consts.WorkloadProviderGrove && (provider != consts.WorkloadProviderComponent || dgd.HasAnyMultinodeComponent()) {
		return "live power-cap updates require the grove workload provider or the component provider without multinode components", false
	}
	for i := range dgd.Spec.Components {
		if dgd.Spec.Components[i].ComponentType == consts.ComponentTypePlanner {
			return "live power-cap updates are unsupported for graphs with a planner component", false
		}
	}

	// LPX has a separate worker lifecycle and is outside this feature's provider scope.
	if dgd.HasLPXComponent() {
		return "live power-cap updates are unsupported for graphs with LPX components", false
	}

	// Keep inherited caps in the workload templates, with their existing precedence.
	if _, inherited := dgd.Spec.Annotations[consts.KubeAnnotationGPUPowerLimit]; inherited {
		return "live power-cap updates are unsupported when spec.annotations sets " + consts.KubeAnnotationGPUPowerLimit, false
	}

	// Preserved v1alpha1 service annotations render into templates outside the component pod-template source.
	// Sorted names make the admission message deterministic.
	if alpha := getDGDAlpha(dgd); alpha != nil {
		for _, name := range slices.Sorted(maps.Keys(alpha.Spec.Services)) {
			service := alpha.Spec.Services[name]
			if service == nil {
				continue
			}
			if _, inherited := service.Annotations[consts.KubeAnnotationGPUPowerLimit]; inherited {
				return "live power-cap updates are unsupported when the v1alpha1 annotations of service " + name + " set " + consts.KubeAnnotationGPUPowerLimit, false
			}
		}
	}

	return "", true
}

// DesiredWorkerPowerCaps returns each worker's explicit pod-template cap, bounded to supported graphs.
// dgd must be non-nil.
func DesiredWorkerPowerCaps(dgd *v1beta1.DynamoGraphDeployment) map[string]string {
	var caps map[string]string
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if IsWorkerComponent(string(component.ComponentType)) && component.PodTemplate != nil && !HasRolePodTemplates(component) {
			if cap, ok := component.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit]; ok {
				if caps == nil {
					if _, supported := LivePowerCapSupport(dgd); !supported {
						return nil
					}
					caps = make(map[string]string)
				}
				caps[component.ComponentName] = cap
			}
		}
	}
	return caps
}
