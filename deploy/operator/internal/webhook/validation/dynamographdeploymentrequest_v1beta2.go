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
	"strings"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/aisimulate"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// parametersPath is where the native AISimulate input lives in a v1beta2 Request.
var parametersPath = field.NewPath("spec", "aisimulate", "parameters")

// validateDynamoGraphDeploymentRequestEnvelopeV1beta2 validates the v1beta2 spec carried in
// spec.v1beta2 of a v1beta1 object. raw is that field and fldPath is its path; a spec that
// does not decode is reported there. Errors about the decoded spec use the v1beta2 spec's own
// field paths (spec.model, spec.aisimulate, ...), which are what the user wrote, so they are
// rooted at spec rather than at fldPath.
func validateDynamoGraphDeploymentRequestEnvelopeV1beta2(
	raw *apiextensionsv1.JSON,
	fldPath *field.Path,
) field.ErrorList {
	var spec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(raw.Raw, &spec); err != nil {
		return field.ErrorList{field.Invalid(fldPath, "<redacted>", "must be a valid v1beta2 spec: "+err.Error())}
	}
	return validateDynamoGraphDeploymentRequestV1beta2(&spec)
}

// validateDynamoGraphDeploymentRequestV1beta2 validates a defaulted v1beta2 Request spec: the
// typed fields, that spec.aisimulate.parameters is an object, and that the effective AISimulate
// input names the hardware. The rest of the parameters content is not validated here; the
// AISimulate input schema owns it. Update calls this only when the spec changed, so objects
// stored before this check keep working until they are edited.
func validateDynamoGraphDeploymentRequestV1beta2(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) field.ErrorList {
	errs := validateDynamoGraphDeploymentRequestSpecV1beta2(spec, field.NewPath("spec"))
	if _, err := aisimulate.Parameters(spec); err != nil {
		errs = append(errs, field.Invalid(parametersPath, "<redacted>", err.Error()))
	} else {
		errs = append(errs, validateEffectiveHardwareV1beta2(spec)...)
	}
	return errs
}

// validateDynamoGraphDeploymentRequestSpecV1beta2 validates spec. spec and fldPath must not be nil.
func validateDynamoGraphDeploymentRequestSpecV1beta2(
	spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
) field.ErrorList {
	var errs field.ErrorList
	errs = append(errs, validateModelSpecV1beta2(spec.Model, fldPath.Child("model"))...)
	errs = append(errs, validateBackendsV1beta2(spec.Backends, fldPath.Child("backends"))...)
	if spec.Workload != nil {
		errs = append(errs, validateWorkloadSpecV1beta2(spec.Workload, fldPath.Child("workload"))...)
	}
	errs = append(errs, validateAISimulateSpecV1beta2(spec.AISimulate, fldPath.Child("aisimulate"))...)
	return errs
}

func validateModelSpecV1beta2(model *nvidiacomv1beta2.ModelSpec, fldPath *field.Path) field.ErrorList {
	if model == nil {
		return field.ErrorList{field.Required(fldPath, "model is required")}
	}
	if strings.TrimSpace(model.Name) == "" {
		return field.ErrorList{field.Required(fldPath.Child("name"), "must not be blank")}
	}
	return nil
}

func validateBackendsV1beta2(backends []nvidiacomv1beta2.Backend, fldPath *field.Path) field.ErrorList {
	var errs field.ErrorList
	seen := make(map[nvidiacomv1beta2.Backend]bool, len(backends))
	for i, backend := range backends {
		if seen[backend] {
			errs = append(errs, field.Duplicate(fldPath.Index(i), backend))
		}
		seen[backend] = true
	}
	return errs
}

func validateWorkloadSpecV1beta2(workload *nvidiacomv1beta2.WorkloadSpec, fldPath *field.Path) field.ErrorList {
	if workload.Static == nil {
		return nil
	}
	return validateStaticWorkloadSpecV1beta2(workload.Static, fldPath.Child("static"))
}

func validateStaticWorkloadSpecV1beta2(static *nvidiacomv1beta2.StaticWorkloadSpec, fldPath *field.Path) field.ErrorList {
	// The schema bounds each ratio to [0, 1] and the number of entries to one or two.
	if len(static.KVLoadRatio) == 2 && static.KVLoadRatio[0] >= static.KVLoadRatio[1] {
		return field.ErrorList{field.Invalid(fldPath.Child("kvLoadRatio"), static.KVLoadRatio, "a range needs min < max")}
	}
	return nil
}

// validateAISimulateSpecV1beta2 validates the execution artifact. Admission persists the
// operator's default image, so a Request that still has none was created while the operator
// had no default image configured. spec may be nil.
func validateAISimulateSpecV1beta2(spec *nvidiacomv1beta2.AISimulateSpec, fldPath *field.Path) field.ErrorList {
	if spec == nil || spec.Image == "" {
		return field.ErrorList{field.Required(fldPath.Child("image"),
			"an AISimulate image is required; set it, or configure the operator's default AISimulate image")}
	}
	if _, err := aisimulate.ResolveVersion(spec.Image, spec.VersionOverride); err != nil {
		return field.ErrorList{field.Required(fldPath.Child("versionOverride"),
			"is required when spec.aisimulate.image has no parseable version tag")}
	}
	return nil
}

// validateEffectiveHardwareV1beta2 requires the AISimulate input that a Run would execute to name
// the GPU SKU and budget. The input is the typed spec lowered and then patched by
// spec.aisimulate.parameters, so hardware given only in the parameters satisfies it, and a
// parameters null that deletes discovered hardware does not. Errors are reported at the typed
// spec.hardware.gpu fields, which is where the user fixes them.
func validateEffectiveHardwareV1beta2(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) field.ErrorList {
	input, err := aisimulate.Lower(spec)
	if err != nil {
		return field.ErrorList{field.Invalid(parametersPath, "<redacted>", err.Error())}
	}
	gpuPath := field.NewPath("spec", "hardware", "gpu")
	var errs field.ErrorList
	if !effectiveValuePresent(input, aisimulate.SearchSpaceHardwareSKU) {
		errs = append(errs, field.Required(gpuPath.Child("sku"),
			"the effective AISimulate input has no hardware SKU: set it, enable GPU discovery, or set spec.aisimulate.parameters.search_space.hardware_sku"))
	}
	if !effectiveValuePresent(input, aisimulate.SearchSpaceGPUBudget) {
		errs = append(errs, field.Required(gpuPath.Child("budget"),
			"the effective AISimulate input has no GPU budget: set it, enable GPU discovery, or set spec.aisimulate.parameters.search_space.gpu_budget"))
	}
	return errs
}

// effectiveValuePresent reports whether input holds a non-null, non-empty value at path.
func effectiveValuePresent(input map[string]any, path []string) bool {
	value, found, err := unstructured.NestedFieldNoCopy(input, path...)
	if err != nil || !found || value == nil {
		return false
	}
	text, isString := value.(string)
	return !isString || text != ""
}
