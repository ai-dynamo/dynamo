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
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func v1beta2Details(errs field.ErrorList) string {
	var out []string
	for _, err := range errs {
		out = append(out, err.Field+": "+err.Detail)
	}
	return strings.Join(out, "\n")
}

func v1beta2RequireError(t *testing.T, errs field.ErrorList, substr string) {
	t.Helper()
	if !strings.Contains(v1beta2Details(errs), substr) {
		t.Fatalf("want an error containing %q, got:\n%s", substr, v1beta2Details(errs))
	}
}

func v1beta2RequireNoError(t *testing.T, errs field.ErrorList) {
	t.Helper()
	if len(errs) > 0 {
		t.Fatalf("unexpected errors:\n%s", v1beta2Details(errs))
	}
}

func v1beta2Envelope(raw string) *apiextensionsv1.JSON {
	return &apiextensionsv1.JSON{Raw: []byte(raw)}
}

func TestValidateEnvelopeV1beta2(t *testing.T) {
	path := field.NewPath("spec", "v1beta2")

	v1beta2RequireNoError(t, validateDynamoGraphDeploymentRequestEnvelopeV1beta2(
		v1beta2Envelope(`{"model":{"name":"meta-llama/Llama-3.1-8B-Instruct"}}`), path))

	v1beta2RequireError(t, validateDynamoGraphDeploymentRequestEnvelopeV1beta2(v1beta2Envelope(`{"model": 3}`), path), "spec.v1beta2: must be a valid v1beta2 spec")

	v1beta2RequireError(t, validateDynamoGraphDeploymentRequestEnvelopeV1beta2(v1beta2Envelope(`{
		"model":{"name":"meta-llama/Llama-3.1-8B-Instruct"},
		"objective":{"mode":"Pareto","metrics":["Throughput","TTFT"]}
	}`), path), "cannot be lowered to the public AISimulate recommendation input")
}
