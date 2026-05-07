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
	hash, err := v1alpha1.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		// Fallback to empty hash on error (shouldn't happen with valid input)
		return "00000000"
	}
	return hash
}
