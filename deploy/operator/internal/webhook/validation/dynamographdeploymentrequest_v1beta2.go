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

// validateDynamoGraphDeploymentRequestSpecV1beta2 validates the typed rules that the source
// schema and CEL cannot express. The generated OpenAPI schema and CEL of the v1beta2 API have
// already run before conversion reaches this webhook, so they are not repeated here: native
// model presence, the model name length, backend uniqueness and the kvLoadRatio range order.
// spec and fldPath must not be nil.
func validateDynamoGraphDeploymentRequestSpecV1beta2(
	spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
) field.ErrorList {
	return validateAISimulateSpecV1beta2(spec.AISimulate, fldPath.Child("aisimulate"))
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
		if spec.VersionOverride != "" {
			return field.ErrorList{field.Invalid(fldPath.Child("versionOverride"), spec.VersionOverride,
				"must be a semantic version such as \"0.13.0\"")}
		}
		return field.ErrorList{field.Required(fldPath.Child("versionOverride"),
			"is required when the tag of spec.aisimulate.image is not a Dynamo release with a bundled AISimulate version")}
	}
	return nil
}
