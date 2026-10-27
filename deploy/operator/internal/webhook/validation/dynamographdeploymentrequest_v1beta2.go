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
func validateDynamoGraphDeploymentRequestEnvelopeV1beta2(
	raw *apiextensionsv1.JSON,
	fldPath *field.Path,
) field.ErrorList {
	var spec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(raw.Raw, &spec); err != nil {
		return field.ErrorList{field.Invalid(fldPath, "<redacted>", "must be a valid v1beta2 spec: "+err.Error())}
	}

	allErrs := validateDynamoGraphDeploymentRequestSpecV1beta2(&spec, field.NewPath("spec"))
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
func validateDynamoGraphDeploymentRequestSpecV1beta2(
	spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	fldPath *field.Path,
) field.ErrorList {
	if spec.AISimulate == nil {
		return nil
	}
	return validateAISimulateSpec(spec.AISimulate, fldPath.Child("aisimulate"))
}
