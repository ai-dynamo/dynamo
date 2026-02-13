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
)

// ComputeDGDWorkersSpecHash computes a deterministic hash of all worker service specs.
//
// The hash uses an exclusion-based approach: the entire DynamoComponentDeploymentSharedSpec
// is hashed after zeroing out fields that do NOT affect the pod template. This ensures
// that any new field added to the spec triggers a rolling update by default (safe by
// default), and only explicitly excluded fields are ignored.
//
// Excluded fields (do not affect the pod template):
//   - ServiceName, ComponentType, SubComponentType: identity fields
//   - DynamoNamespace: deprecated, not used in pod spec generation
//   - Replicas: scaling, not pod template
//   - Autoscaling: deprecated, ignored
//   - ScalingAdapter: scaling configuration, not pod template
//   - Ingress: networking resources, not pod template
//   - ModelRef: headless service creation, not pod template
//   - EPPConfig: EPP-only, not applicable to workers
//   - Annotations, Labels: applied to K8s resources, not pod template
//     (pod-level metadata is in ExtraPodMetadata which IS included)
//
// Only worker components (prefill, decode, worker) are included in the hash.
func ComputeDGDWorkersSpecHash(dgd *v1alpha1.DynamoGraphDeployment) string {
	// Collect worker specs in sorted order for deterministic hashing
	var workerNames []string
	for name, spec := range dgd.Spec.Services {
		if spec != nil && IsWorkerComponent(spec.ComponentType) {
			workerNames = append(workerNames, name)
		}
	}
	sort.Strings(workerNames)

	// Build hash input map (sorted keys for determinism)
	hashInputs := make(map[string]v1alpha1.DynamoComponentDeploymentSharedSpec)
	for _, name := range workerNames {
		spec := dgd.Spec.Services[name]
		hashInputs[name] = stripNonPodTemplateFields(spec)
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

// stripNonPodTemplateFields returns a copy of the spec with fields that do NOT affect
// the pod template zeroed out. The remaining fields are all pod-template-affecting and
// will be included in the hash.
//
// This is an exclusion-based approach: new fields added to DynamoComponentDeploymentSharedSpec
// are included in the hash by default. Only fields explicitly listed here are excluded.
func stripNonPodTemplateFields(spec *v1alpha1.DynamoComponentDeploymentSharedSpec) v1alpha1.DynamoComponentDeploymentSharedSpec {
	// Start with a shallow copy of the full spec
	stripped := *spec

	// Zero out fields that do NOT affect the pod template.
	// These are identity, scaling, networking, and metadata fields.
	stripped.Annotations = nil
	stripped.Labels = nil
	stripped.ServiceName = ""
	stripped.ComponentType = ""
	stripped.SubComponentType = ""
	stripped.DynamoNamespace = nil
	stripped.Replicas = nil
	stripped.Autoscaling = nil //nolint:staticcheck // SA1019: intentionally clearing deprecated field for backward compatibility
	stripped.ScalingAdapter = nil
	stripped.Ingress = nil
	stripped.ModelRef = nil
	stripped.EPPConfig = nil

	return stripped
}
