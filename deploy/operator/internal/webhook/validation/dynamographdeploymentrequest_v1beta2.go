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
	"k8s.io/apimachinery/pkg/util/validation/field"
)

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
// typed fields and the image and version selection. spec.aisimulate.parameters is opaque here: the
// CRD types it as an object, and the validation of its content belongs to the schema of the
// AISimulate release the Request selects.
func validateDynamoGraphDeploymentRequestV1beta2(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) field.ErrorList {
	return validateDynamoGraphDeploymentRequestSpecV1beta2(spec, field.NewPath("spec"))
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
