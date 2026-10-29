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

package dgdrinput

import (
	"fmt"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateSpec runs every admission check on a v1beta2 Request spec: the typed objective
// fields against spec.aisimulate.parameters, then the complete effective input.
func ValidateSpec(spec *v1beta2.DynamoGraphDeploymentRequestSpec, opts Options) field.ErrorList {
	errs := Conflicts(spec)
	input, err := Build(spec, opts)
	if err != nil {
		return append(errs, field.Invalid(field.NewPath("spec", "aisimulate", "parameters"), "<redacted>", err.Error()))
	}
	return append(errs, Validate(input)...)
}

// NewViolations returns the errors in current that previous did not already have. An update
// is judged on what it introduces, so a Request that was admitted earlier can still be
// edited, scaled or cancelled while it carries an older violation. Errors are matched by
// field, type and detail; the offending value is ignored.
func NewViolations(current, previous field.ErrorList) field.ErrorList {
	seen := make(map[string]struct{}, len(previous))
	for _, e := range previous {
		seen[violationKey(e)] = struct{}{}
	}
	var out field.ErrorList
	for _, e := range current {
		if _, ok := seen[violationKey(e)]; !ok {
			out = append(out, e)
		}
	}
	return out
}

func violationKey(e *field.Error) string {
	return fmt.Sprintf("%s|%s|%s", e.Field, e.Type, e.Detail)
}
