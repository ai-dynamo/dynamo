/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package v1beta1

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const dgdWorkerHashInputVersion = "dgd-worker-rollout-v2"

type dgdWorkerHashInput struct {
	Version          string                              `json:"version"`
	BackendFramework string                              `json:"backendFramework,omitempty"`
	Env              []corev1.EnvVar                     `json:"env,omitempty"`
	Workers          map[string]workerComponentHashInput `json:"workers,omitempty"`
}

type workerComponentHashInput struct {
	GlobalDynamoNamespace bool                    `json:"globalDynamoNamespace,omitempty"`
	PodTemplate           *corev1.PodTemplateSpec `json:"podTemplate,omitempty"`
	Multinode             *MultinodeSpec          `json:"multinode,omitempty"`
	SharedMemorySize      *resource.Quantity      `json:"sharedMemorySize,omitempty"`
	FrontendSidecar       *string                 `json:"frontendSidecar,omitempty"`
	CompilationCache      *CompilationCacheConfig `json:"compilationCache,omitempty"`
	Experimental          *ExperimentalSpec       `json:"experimental,omitempty"`
}

// ComputeDGDWorkersSpecHash computes a deterministic hash of all v1beta1
// worker component specs.
//
// The v2 hash uses a versioned, normalized input made from fields that affect
// generated worker pods. This avoids tying rollout decisions to the wire shape
// of whichever CRD version the API server returns.
//
// Excluded fields (do not affect the pod template):
//   - ComponentType: identity field
//   - Replicas: scaling, not pod template
//   - ScalingAdapter: scaling configuration, not pod template
//   - ModelRef: headless service creation, not pod template
//   - EPPConfig: EPP-only, not applicable to workers
//   - TopologyConstraint: scheduler placement, not pod template content
//
// Only worker components (prefill, decode, worker) are included in the hash.
func ComputeDGDWorkersSpecHash(dgd *DynamoGraphDeployment) (string, error) {
	if dgd == nil {
		return "", fmt.Errorf("nil DynamoGraphDeployment")
	}

	var workerNames []string
	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		if isWorkerComponent(spec.ComponentType) {
			workerNames = append(workerNames, spec.ComponentName)
		}
	}
	sort.Strings(workerNames)

	hashInput := dgdWorkerHashInput{
		Version:          dgdWorkerHashInputVersion,
		BackendFramework: dgd.Spec.BackendFramework,
		Env:              dgd.Spec.Env,
		Workers:          make(map[string]workerComponentHashInput, len(workerNames)),
	}
	for _, name := range workerNames {
		spec := dgd.GetComponentByName(name)
		if spec != nil {
			hashInput.Workers[name] = workerHashInputForSpec(spec)
		}
	}

	data, err := json.Marshal(hashInput)
	if err != nil {
		return "", fmt.Errorf("marshal DGD worker hash input: %w", err)
	}

	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])[:8], nil
}

func isWorkerComponent(componentType ComponentType) bool {
	return componentType == ComponentTypeWorker ||
		componentType == ComponentTypePrefill ||
		componentType == ComponentTypeDecode
}

func workerHashInputForSpec(spec *DynamoComponentDeploymentSharedSpec) workerComponentHashInput {
	stripped := stripNonPodTemplateFields(spec)
	return workerComponentHashInput{
		GlobalDynamoNamespace: stripped.GlobalDynamoNamespace,
		PodTemplate:           stripped.PodTemplate,
		Multinode:             stripped.Multinode,
		SharedMemorySize:      stripped.SharedMemorySize,
		FrontendSidecar:       stripped.FrontendSidecar,
		CompilationCache:      stripped.CompilationCache,
		Experimental:          stripped.Experimental,
	}
}

// stripNonPodTemplateFields returns a copy of the beta spec with fields that do
// not affect worker pod contents zeroed out. The v2 hash then selects its
// explicit canonical input from this normalized copy.
func stripNonPodTemplateFields(spec *DynamoComponentDeploymentSharedSpec) DynamoComponentDeploymentSharedSpec {
	stripped := *spec

	stripped.ComponentType = ""
	stripped.Replicas = nil
	stripped.ScalingAdapter = nil
	stripped.ModelRef = nil
	stripped.EPPConfig = nil
	stripped.TopologyConstraint = nil
	if stripped.PodTemplate != nil {
		stripped.PodTemplate = stripped.PodTemplate.DeepCopy()
		if reflect.DeepEqual(*stripped.PodTemplate, corev1.PodTemplateSpec{}) {
			stripped.PodTemplate = nil
		}
	}

	return stripped
}
