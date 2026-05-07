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

package dynamo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"reflect"
	"sort"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
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
	GlobalDynamoNamespace bool                            `json:"globalDynamoNamespace,omitempty"`
	PodTemplate           *corev1.PodTemplateSpec         `json:"podTemplate,omitempty"`
	Multinode             *v1beta1.MultinodeSpec          `json:"multinode,omitempty"`
	SharedMemorySize      *resource.Quantity              `json:"sharedMemorySize,omitempty"`
	FrontendSidecar       *string                         `json:"frontendSidecar,omitempty"`
	CompilationCache      *v1beta1.CompilationCacheConfig `json:"compilationCache,omitempty"`
	Experimental          *v1beta1.ExperimentalSpec       `json:"experimental,omitempty"`
}

// ComputeV1beta1DGDWorkersSpecHash computes a deterministic hash of all v1beta1 worker component specs.
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
func ComputeV1beta1DGDWorkersSpecHash(dgd *v1beta1.DynamoGraphDeployment) string {
	var workerNames []string
	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		if IsWorkerComponent(string(spec.ComponentType)) {
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
		return "00000000"
	}

	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])[:8]
}

func workerHashInputForSpec(spec *v1beta1.DynamoComponentDeploymentSharedSpec) workerComponentHashInput {
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

// ComputeLegacyAlphaDGDWorkersSpecHash returns the v1alpha1 worker hash used by
// pre-v1beta1 controllers. Converted v1alpha1 DGDs carry the exact old hash in
// an annotation; the ConvertFrom fallback is only for synthetic beta objects.
func ComputeLegacyAlphaDGDWorkersSpecHash(dgd *v1beta1.DynamoGraphDeployment) (string, error) {
	if hash := v1alpha1.GetDGDLegacyWorkerHash(dgd); hash != "" {
		return hash, nil
	}
	alpha := &v1alpha1.DynamoGraphDeployment{}
	if err := alpha.ConvertFrom(dgd); err != nil {
		return "", err
	}
	return v1alpha1.ComputeDGDWorkersSpecHash(alpha)
}

func ClearLegacyAlphaDGDWorkersSpecHash(dgd *v1beta1.DynamoGraphDeployment) {
	v1alpha1.ClearDGDLegacyWorkerHash(dgd)
}

// stripNonPodTemplateFields returns a copy of the beta spec with fields that do
// not affect worker pod contents zeroed out. The v2 hash then selects its
// explicit canonical input from this normalized copy.
func stripNonPodTemplateFields(spec *v1beta1.DynamoComponentDeploymentSharedSpec) v1beta1.DynamoComponentDeploymentSharedSpec {
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
