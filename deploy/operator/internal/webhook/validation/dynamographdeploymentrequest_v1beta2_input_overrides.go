/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package validation

import (
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/aisimulate"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateAISimulateSpec validates the Request's native AISimulate overrides.
func validateAISimulateSpec(
	input *nvidiacomv1beta2.AISimulateSpec,
	fldPath *field.Path,
) field.ErrorList {
	// Use the lowering package's decoder as the single interpretation of RawExtension.
	overrides, err := aisimulate.InputOverrides(&nvidiacomv1beta2.NativeSearchInput{AISimulate: input})
	if err != nil || overrides == nil {
		// The lowering validation reports malformed override documents once.
		return nil
	}

	dgdrRunValue, found := overrides["dgdr_run"]
	if !found {
		return nil
	}
	dgdrRun, ok := dgdrRunValue.(map[string]any)
	if !ok {
		return field.ErrorList{field.Forbidden(
			fldPath.Child("inputOverrides", "dgdr_run"),
			"must be an object because the operator owns required execution fields within it",
		)}
	}

	// Search overrides remain authoritative; only execution values coupled to the
	// controller-owned Job and shared volume are reserved.
	reserved := []string{
		"derive_runtime_images",
		"max_candidates",
		"name",
		"namespace",
		"num_gpus_per_node",
		"runtime_image",
		"runtime_version_override",
		"snapshot_dir",
	}
	allErrs := field.ErrorList{}
	for _, key := range reserved {
		if _, found := dgdrRun[key]; found {
			allErrs = append(allErrs, field.Forbidden(
				fldPath.Child("inputOverrides", "dgdr_run", key),
				"is managed by the operator and cannot be overridden",
			))
		}
	}
	return allErrs
}
