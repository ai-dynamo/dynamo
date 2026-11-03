// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
)

// SupportsLivePowerCaps reports the Stage-1 provider and Planner boundary. dgd must be non-nil.
func SupportsLivePowerCaps(dgd *v1beta1.DynamoGraphDeployment) bool {
	provider := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]
	if provider != "grove" && (provider != "component" || dgd.HasAnyMultinodeComponent()) {
		return false
	}
	for i := range dgd.Spec.Components {
		if dgd.Spec.Components[i].ComponentType == consts.ComponentTypePlanner {
			return false
		}
	}
	return true
}

// WorkerSpecsEqual compares the same non-scaling spec used by worker hashing.
// Callers align identity and the explicitly retained cap before comparison; neither input is mutated.
func WorkerSpecsEqual(a, b *v1beta1.DynamoComponentDeployment) bool {
	return apiequality.Semantic.DeepEqual(workerHashSpec(a), workerHashSpec(b))
}

// GroveWorkerIdentity selects an observed generation before names and runtime bindings are rendered.
// An empty Hash preserves a legacy unsuffixed workload. PowerCaps contains creation-template values.
type GroveWorkerIdentity struct {
	Hash      string
	PowerCaps map[string]string
}
