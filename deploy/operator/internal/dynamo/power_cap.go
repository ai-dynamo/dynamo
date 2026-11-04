// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"maps"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

// LivePowerCapSupport reports the Stage-1 graph boundary and, when excluded, why. dgd must be non-nil.
func LivePowerCapSupport(dgd *v1beta1.DynamoGraphDeployment) (reason string, supported bool) {
	provider := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]
	if provider != "grove" && (provider != "component" || dgd.HasAnyMultinodeComponent()) {
		return "live power-cap updates require the grove workload provider or the component provider without multinode components", false
	}
	for i := range dgd.Spec.Components {
		if dgd.Spec.Components[i].ComponentType == consts.ComponentTypePlanner {
			return "live power-cap updates are unsupported for graphs with a planner component", false
		}
	}

	// LPX serving identity derives from the edited worker hash and cannot retain an old cap.
	if dgd.HasLPXComponent() {
		return "live power-cap updates are unsupported for graphs with LPX components", false
	}

	// Inherited caps have no per-worker source that preservation and propagation can follow.
	if _, inherited := dgd.Spec.Annotations[consts.KubeAnnotationGPUPowerLimit]; inherited {
		return "live power-cap updates are unsupported when spec.annotations sets " + consts.KubeAnnotationGPUPowerLimit, false
	}
	return "", true
}

// WorkerSpecsEqual compares the non-scaling spec and resolved runtime versions used by worker hashing.
// Callers align identity and the explicitly retained cap before comparison; neither input is mutated.
func WorkerSpecsEqual(a, b *v1beta1.DynamoComponentDeployment) bool {
	return resolvedRuntimeVersionForHash(&a.Spec.DynamoComponentDeploymentSharedSpec) == resolvedRuntimeVersionForHash(&b.Spec.DynamoComponentDeploymentSharedSpec) &&
		maps.Equal(resolvedRoleRuntimeVersionsForHash(&a.Spec.DynamoComponentDeploymentSharedSpec), resolvedRoleRuntimeVersionsForHash(&b.Spec.DynamoComponentDeploymentSharedSpec)) &&
		apiequality.Semantic.DeepEqual(workerHashSpec(a), workerHashSpec(b))
}

// GroveWorkerIdentity selects an observed generation before names and runtime bindings are rendered.
// An empty Hash preserves a legacy unsuffixed workload. PowerCaps contains creation-template values.
type GroveWorkerIdentity struct {
	Hash      string
	PowerCaps map[string]string
}
