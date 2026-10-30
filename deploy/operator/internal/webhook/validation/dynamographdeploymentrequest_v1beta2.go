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
	"encoding/json"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/aisimulate"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateDynamoGraphDeploymentRequestEnvelopeV1beta2 validates the v1beta2 spec carried in
// spec.v1beta2 of a v1beta1 object. raw is that field and fldPath is its path; a spec that
// does not decode is reported there. Errors about the decoded spec use the v1beta2 spec's own
// field paths (spec.model, spec.aisimulate, ...), which are what the user wrote, so they are
// rooted at spec rather than at fldPath.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestEnvelopeV1beta2(
	raw *apiextensionsv1.JSON,
	fldPath *field.Path,
) field.ErrorList {
	var spec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(raw.Raw, &spec); err != nil {
		return field.ErrorList{field.Invalid(fldPath, "<redacted>", "must be a valid v1beta2 spec: "+err.Error())}
	}

	allErrs := v.validateDynamoGraphDeploymentRequestSpecV1beta2(&spec, field.NewPath("spec"))
	if _, err := aisimulate.Lower(&spec.NativeSearchInput); err != nil {
		allErrs = append(allErrs, field.Invalid(
			field.NewPath("spec"),
			"<redacted>",
			"cannot be lowered to the public AISimulate recommendation input: "+err.Error(),
		))
	}
	return allErrs
}

// validateDynamoGraphDeploymentRequestSpecV1beta2 validates spec. spec and fldPath must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateDynamoGraphDeploymentRequestSpecV1beta2(
	spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
) field.ErrorList {
	allErrs := field.ErrorList{}
	if spec.Hardware != nil {
		allErrs = append(allErrs, v.validateHardwareSpecV1beta2(spec.Hardware, fldPath.Child("hardware"))...)
	}
	if spec.AISimulate != nil {
		allErrs = append(allErrs, validateAISimulateSpec(spec.AISimulate, fldPath.Child("aisimulate"))...)
	}
	if spec.Overrides != nil {
		allErrs = append(allErrs, v.validateOverridesSpecV1beta2(spec.Overrides, fldPath.Child("overrides"))...)
	}
	return allErrs
}

// validateOverridesSpecV1beta2 validates the partial resource envelope. spec and fldPath must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateOverridesSpecV1beta2(
	spec *nvidiacomv1beta2.OverridesSpec,
	fldPath *field.Path,
) field.ErrorList {
	if spec.DGD == nil {
		return nil
	}

	// Nested DGD values are checked on the complete generated deployment in the publisher.
	var override map[string]json.RawMessage
	path := fldPath.Child("dgd")
	if err := json.Unmarshal(spec.DGD.Raw, &override); err != nil || override == nil {
		return field.ErrorList{field.Invalid(path, "<redacted>", "must be a partial DGD object")}
	}
	allErrs := field.ErrorList{}
	for _, member := range []struct{ name, expected string }{
		{"apiVersion", "nvidia.com/v1beta1"}, {"kind", "DynamoGraphDeployment"},
	} {
		var value string
		if err := json.Unmarshal(override[member.name], &value); err != nil || value != member.expected {
			allErrs = append(allErrs, field.NotSupported(path.Child(member.name), value, []string{member.expected}))
		}
	}
	if _, found := override["status"]; found {
		allErrs = append(allErrs, field.Forbidden(path.Child("status"), "DGD status cannot be overridden"))
	}
	return allErrs
}

// validateHardwareSpecV1beta2 validates hardware. spec and fldPath must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateHardwareSpecV1beta2(
	spec *nvidiacomv1beta2.HardwareSpec,
	fldPath *field.Path,
) field.ErrorList {
	if spec.GPU == nil {
		return nil
	}
	return v.validateGPUHardwareSpecV1beta2(spec.GPU, fldPath.Child("gpu"))
}

// validateGPUHardwareSpecV1beta2 validates gpu. spec and fldPath must not be nil.
func (v *dynamoGraphDeploymentRequestValidation) validateGPUHardwareSpecV1beta2(
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
