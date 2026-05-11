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

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const dgdWorkerHashInputVersion = "dgd-worker-rollout-v2"

type dgdWorkerHashInput struct {
	Version          string                              `json:"version"`
	BackendFramework string                              `json:"backendFramework,omitempty"`
	Workers          map[string]workerComponentHashInput `json:"workers,omitempty"`
}

type workerComponentHashInput struct {
	ComponentType         ComponentType           `json:"type,omitempty"`
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
// WARNING: this hash is part of the rolling-update compatibility contract. A
// changed value for the same semantic input can roll workers across existing
// deployments. Update the golden test only with an explicit migration plan.
//
// The v2 hash uses a versioned, normalized input made from fields that affect
// generated worker pods. This avoids tying rollout decisions to the wire shape
// of whichever CRD version the API server returns.
//
// Excluded fields (do not affect worker pod contents):
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
		Workers:          make(map[string]workerComponentHashInput, len(workerNames)),
	}
	for _, name := range workerNames {
		spec := dgd.GetComponentByName(name)
		if spec != nil {
			hashInput.Workers[name] = workerHashInputForSpec(spec, dgd)
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

func workerHashInputForSpec(spec *DynamoComponentDeploymentSharedSpec, dgd *DynamoGraphDeployment) workerComponentHashInput {
	stripped := stripNonPodTemplateFields(spec)
	applyDGDSpecDefaultsToWorkerHashInput(&stripped, spec.ComponentType, dgd)
	return workerComponentHashInput{
		ComponentType:         spec.ComponentType,
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

func applyDGDSpecDefaultsToWorkerHashInput(
	component *DynamoComponentDeploymentSharedSpec,
	componentType ComponentType,
	dgd *DynamoGraphDeployment,
) {
	if component == nil || dgd == nil {
		return
	}

	if len(dgd.Spec.Annotations) > 0 {
		podTemplate := ensureWorkerHashPodTemplate(component)
		podTemplate.Annotations = mergeLowPriorityWorkerHashMetadata(podTemplate.Annotations, dgd.Spec.Annotations)
	}
	if len(dgd.Spec.Labels) > 0 {
		podTemplate := ensureWorkerHashPodTemplate(component)
		podTemplate.Labels = mergeLowPriorityWorkerHashMetadata(podTemplate.Labels, dgd.Spec.Labels)
	}
	if dgd.HasEPPComponent() && isWorkerComponent(componentType) {
		podTemplate := ensureWorkerHashPodTemplate(component)
		if podTemplate.Labels == nil {
			podTemplate.Labels = map[string]string{}
		}
		podTemplate.Labels[commonconsts.KubeLabelDynamoComponentClass] = commonconsts.ComponentClassWorker
	}
	if len(dgd.Spec.Env) > 0 {
		podTemplate := ensureWorkerHashPodTemplate(component)
		main := ensureWorkerHashMainContainer(podTemplate)
		main.Env = mergeWorkerHashEnvs(dgd.Spec.Env, main.Env)
	}
	if component.PodTemplate != nil && reflect.DeepEqual(*component.PodTemplate, corev1.PodTemplateSpec{}) {
		component.PodTemplate = nil
	}
}

func ensureWorkerHashPodTemplate(component *DynamoComponentDeploymentSharedSpec) *corev1.PodTemplateSpec {
	if component.PodTemplate == nil {
		component.PodTemplate = &corev1.PodTemplateSpec{}
	}
	return component.PodTemplate
}

func ensureWorkerHashMainContainer(podTemplate *corev1.PodTemplateSpec) *corev1.Container {
	for i := range podTemplate.Spec.Containers {
		if podTemplate.Spec.Containers[i].Name == commonconsts.MainContainerName {
			return &podTemplate.Spec.Containers[i]
		}
	}
	podTemplate.Spec.Containers = append([]corev1.Container{{Name: commonconsts.MainContainerName}}, podTemplate.Spec.Containers...)
	return &podTemplate.Spec.Containers[0]
}

func mergeLowPriorityWorkerHashMetadata(dst, src map[string]string) map[string]string {
	if len(src) == 0 {
		return dst
	}
	if dst == nil {
		dst = map[string]string{}
	}
	for k, v := range src {
		if _, exists := dst[k]; !exists {
			dst[k] = v
		}
	}
	return dst
}

func mergeWorkerHashEnvs(common, specific []corev1.EnvVar) []corev1.EnvVar {
	envMap := make(map[string]corev1.EnvVar)
	for _, env := range common {
		envMap[env.Name] = env
	}
	for _, env := range specific {
		envMap[env.Name] = env
	}

	names := make([]string, 0, len(envMap))
	for name := range envMap {
		names = append(names, name)
	}
	sort.Strings(names)

	merged := make([]corev1.EnvVar, 0, len(names))
	for _, name := range names {
		merged = append(merged, envMap[name])
	}
	return merged
}
