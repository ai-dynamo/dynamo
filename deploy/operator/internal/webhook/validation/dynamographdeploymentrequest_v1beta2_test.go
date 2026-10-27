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
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func v1beta2Ptr[T any](v T) *T { return &v }

// v1beta2DefaultedSpec is a Request as it looks once the schema and admission have defaulted it.
func v1beta2DefaultedSpec() *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	metric := nvidiacomv1beta2.ObjectiveMetric("ThroughputPerGPU")
	return &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    &nvidiacomv1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM, nvidiacomv1beta2.BackendSGLang, nvidiacomv1beta2.BackendTRTLLM},
		Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: v1beta2Ptr("h200_sxm"), Budget: v1beta2Ptr(int32(8))}},
		Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
			ISL: v1beta2Ptr(int32(4000)), OSL: v1beta2Ptr(int32(1000)), KVLoadRatio: nvidiacomv1beta2.KVLoadRatio{0, 1},
		}},
		Objective: &nvidiacomv1beta2.ObjectiveSpec{
			Mode:   nvidiacomv1beta2.ObjectiveModeOptimize,
			Metric: &metric,
			SLA:    &nvidiacomv1beta2.ObjectiveSLASpec{TTFTMs: v1beta2Ptr(2000.0), ITLMs: v1beta2Ptr(30.0)},
		},
		AISimulate:     &nvidiacomv1beta2.AISimulateSpec{Image: "registry.example.com/aisimulate:1.6.0"},
		Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
	}
}

func v1beta2WithParameters(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec, parameters string) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	if spec.AISimulate == nil {
		spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Image: "registry.example.com/aisimulate:1.6.0"}
	}
	spec.AISimulate.Parameters = &runtime.RawExtension{Raw: []byte(parameters)}
	return spec
}

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

func TestValidateV1beta2AcceptsADefaultedRequest(t *testing.T) {
	v1beta2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(v1beta2DefaultedSpec()))
}

// The webhook does not repeat what the generated schema and CEL enforce before conversion:
// native model presence, the model name length, backend uniqueness and the kvLoadRatio range
// order. The operatorenv table proves those errors end to end.
func TestValidateV1beta2LeavesSourceSchemaRulesToTheSchema(t *testing.T) {
	spec := v1beta2DefaultedSpec()
	spec.Model = nil
	spec.Backends = []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM, nvidiacomv1beta2.BackendVLLM}
	spec.Workload.Static.KVLoadRatio = nvidiacomv1beta2.KVLoadRatio{0.9, 0.1}
	v1beta2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(spec))
}

func TestValidateV1beta2ParametersAreOpaque(t *testing.T) {
	for _, parameters := range []string{
		`{"workload":{"isl":0},"goal":{"target":"nope"}}`,
		`[1]`, // the CRD types parameters as an object; admission does not interpret it
	} {
		v1beta2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(
			v1beta2WithParameters(v1beta2DefaultedSpec(), parameters)))
	}
}

func TestValidateV1beta2AISimulateImage(t *testing.T) {
	t.Run("no AISimulate section", func(t *testing.T) {
		spec := v1beta2DefaultedSpec()
		spec.AISimulate = nil
		v1beta2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.aisimulate.image: an AISimulate image is required")
	})
	t.Run("parameters without an image", func(t *testing.T) {
		spec := v1beta2DefaultedSpec()
		spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{}
		v1beta2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.aisimulate.image")
	})
	t.Run("custom tag needs a version", func(t *testing.T) {
		spec := v1beta2DefaultedSpec()
		spec.AISimulate.Image = "registry.example.com/aisimulate:custom"
		v1beta2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.aisimulate.versionOverride: is required when the tag of spec.aisimulate.image is not a Dynamo release with a bundled AISimulate version")
	})
	t.Run("custom tag with a version", func(t *testing.T) {
		spec := v1beta2DefaultedSpec()
		spec.AISimulate.Image = "registry.example.com/aisimulate:custom"
		spec.AISimulate.VersionOverride = "0.13.1"
		v1beta2RequireNoError(t, validateDynamoGraphDeploymentRequestV1beta2(spec))
	})
	t.Run("a Dynamo release without a bundled version needs one", func(t *testing.T) {
		spec := v1beta2DefaultedSpec()
		spec.AISimulate.Image = "registry.example.com/aisimulate:1.2.3"
		spec.AISimulate.VersionOverride = ""
		v1beta2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.aisimulate.versionOverride")
	})
	t.Run("an override that is not a version", func(t *testing.T) {
		spec := v1beta2DefaultedSpec()
		spec.AISimulate.VersionOverride = "bad"
		v1beta2RequireError(t, validateDynamoGraphDeploymentRequestV1beta2(spec), "spec.aisimulate.versionOverride: must be a semantic version")
	})
}

func v1beta2Envelope(raw string) *apiextensionsv1.JSON {
	return &apiextensionsv1.JSON{Raw: []byte(raw)}
}

func TestValidateEnvelopeV1beta2(t *testing.T) {
	path := field.NewPath("spec", "v1beta2")

	good, err := json.Marshal(v1beta2DefaultedSpec())
	if err != nil {
		t.Fatal(err)
	}
	v1beta2RequireNoError(t, validateDynamoGraphDeploymentRequestEnvelopeV1beta2(v1beta2Envelope(string(good)), path))

	bad := v1beta2DefaultedSpec()
	bad.AISimulate = nil
	raw, _ := json.Marshal(bad)
	v1beta2RequireError(t, validateDynamoGraphDeploymentRequestEnvelopeV1beta2(v1beta2Envelope(string(raw)), path), "aisimulate")

	v1beta2RequireError(t, validateDynamoGraphDeploymentRequestEnvelopeV1beta2(v1beta2Envelope(`{"model": 3}`), path), "spec.v1beta2: must be a valid v1beta2 spec")
}
