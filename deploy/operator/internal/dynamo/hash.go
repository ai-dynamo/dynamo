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
	"sort"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

// WorkerPodSpecHashInput contains all fields from worker specs that affect the generated PodSpec.
// This struct is designed to capture any change that would trigger a rolling update in the
// underlying Kubernetes resource (Deployment, PodClique, etc.).
type WorkerPodSpecHashInput struct {
	// ExtraPodSpec contains the full PodSpec and MainContainer overrides
	ExtraPodSpec *v1alpha1.ExtraPodSpec `json:"extraPodSpec,omitempty"`
	// ExtraPodMetadata contains labels and annotations for the pod
	ExtraPodMetadata *v1alpha1.ExtraPodMetadata `json:"extraPodMetadata,omitempty"`
	// Resources contains resource requests and limits
	Resources *v1alpha1.Resources `json:"resources,omitempty"`
	// Envs contains environment variables (sorted for determinism)
	Envs []corev1.EnvVar `json:"envs,omitempty"`
	// EnvFromSecret references a secret for environment variables
	EnvFromSecret *string `json:"envFromSecret,omitempty"`
	// VolumeMounts contains volume mount configuration
	VolumeMounts []v1alpha1.VolumeMount `json:"volumeMounts,omitempty"`
	// SharedMemory contains shared memory configuration
	SharedMemory *v1alpha1.SharedMemorySpec `json:"sharedMemory,omitempty"`
	// LivenessProbe contains liveness probe configuration
	LivenessProbe *corev1.Probe `json:"livenessProbe,omitempty"`
	// ReadinessProbe contains readiness probe configuration
	ReadinessProbe *corev1.Probe `json:"readinessProbe,omitempty"`
	// Multinode contains multinode deployment configuration
	Multinode *v1alpha1.MultinodeSpec `json:"multinode,omitempty"`
}

// ComputeWorkerSpecHash computes a deterministic hash of all worker service specs.
// The hash includes all fields that would trigger a rolling update in the underlying
// Kubernetes resource. This ensures that our traffic shifting logic engages whenever
// the underlying resource would perform a rolling update.
//
// Only worker components (prefill, decode, worker) are included in the hash.
func ComputeWorkerSpecHash(dgd *v1alpha1.DynamoGraphDeployment) string {
	// Collect worker specs in sorted order for deterministic hashing
	var workerNames []string
	for name, spec := range dgd.Spec.Services {
		if spec != nil && IsWorkerComponent(spec.ComponentType) {
			workerNames = append(workerNames, name)
		}
	}
	sort.Strings(workerNames)

	// Build hash input map (sorted keys for determinism)
	hashInputs := make(map[string]WorkerPodSpecHashInput)
	for _, name := range workerNames {
		spec := dgd.Spec.Services[name]
		hashInputs[name] = extractPodSpecHashInput(spec)
	}

	// Serialize to JSON (Go's encoding/json sorts map keys)
	data, err := json.Marshal(hashInputs)
	if err != nil {
		// Fallback to empty hash on error (shouldn't happen with valid input)
		return "00000000"
	}

	// Compute SHA256 and take first 8 characters for readability
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])[:8]
}

// extractPodSpecHashInput extracts all PodSpec-affecting fields from a service spec
func extractPodSpecHashInput(spec *v1alpha1.DynamoComponentDeploymentSharedSpec) WorkerPodSpecHashInput {
	input := WorkerPodSpecHashInput{
		ExtraPodSpec:     spec.ExtraPodSpec,
		ExtraPodMetadata: spec.ExtraPodMetadata,
		Resources:        spec.Resources,
		EnvFromSecret:    spec.EnvFromSecret,
		VolumeMounts:     spec.VolumeMounts,
		SharedMemory:     spec.SharedMemory,
		LivenessProbe:    spec.LivenessProbe,
		ReadinessProbe:   spec.ReadinessProbe,
		Multinode:        spec.Multinode,
	}

	// Sort environment variables by name for deterministic hashing
	if len(spec.Envs) > 0 {
		input.Envs = sortEnvVars(spec.Envs)
	}

	return input
}

// sortEnvVars returns a sorted copy of env vars for deterministic hashing
func sortEnvVars(envs []corev1.EnvVar) []corev1.EnvVar {
	sorted := make([]corev1.EnvVar, len(envs))
	copy(sorted, envs)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})
	return sorted
}

func ComputeDynamoNamespace(dgd *v1alpha1.DynamoGraphDeployment) string {
	return dgd.Namespace + "-" + dgd.Name
}
