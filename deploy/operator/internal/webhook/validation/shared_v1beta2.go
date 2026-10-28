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

// validateNativeSearchInput validates the typed search fields. spec and fldPath must not be nil.
// Schema and CEL own the remaining typed child rules; upstream input validation runs after lowering.
func (v *sharedValidation) validateNativeSearchInput(spec *nvidiacomv1beta2.NativeSearchInput, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.Hardware != nil {
		allErrs = append(allErrs, v.validateHardwareSpecV1beta2(spec.Hardware, fldPath.Child("hardware"))...)
	}
	if spec.AISimulate != nil {
		allErrs = append(allErrs, validateAISimulateSpec(spec.AISimulate, fldPath.Child("aisimulate"))...)
	}
	return allErrs
}

// validateHardwareSpecV1beta2 validates hardware. spec and fldPath must not be nil.
func (v *sharedValidation) validateHardwareSpecV1beta2(
	spec *nvidiacomv1beta2.HardwareSpec,
	fldPath *field.Path,
) field.ErrorList {
	if spec.GPU == nil {
		return nil
	}
	return v.validateGPUHardwareSpecV1beta2(spec.GPU, fldPath.Child("gpu"))
}

// validateGPUHardwareSpecV1beta2 validates gpu. spec and fldPath must not be nil.
func (v *sharedValidation) validateGPUHardwareSpecV1beta2(
	spec *nvidiacomv1beta2.GPUHardwareSpec,
	fldPath *field.Path,
) field.ErrorList {
	if spec.SKU != nil && !aisimulate.IsKnownHardwareSKU(*spec.SKU) {
		v.warnf(
			"%s: %q is not known to the AISimulate release bundled with this operator; the value is passed through unchanged and may work with an external AISimulate systems catalog",
			fldPath.Child("sku"), *spec.SKU,
		)
	}
	return nil
}

// validateAISimulateSpec validates native AISimulate overrides. input and fldPath must not be nil.
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
