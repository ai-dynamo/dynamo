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

package defaulting

import (
	"encoding/json"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/utils/ptr"
)

func structuralTestOptions() dynamoGraphDeploymentRequestV1beta2DefaultOptions {
	return dynamoGraphDeploymentRequestV1beta2DefaultOptions{
		hardware:        hardwareDefaultsV1beta2{sku: "h200_sxm", budget: 8},
		aisimulateImage: "planner:test",
	}
}

func structuralDefaultedSpec() nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	return nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
		Backends: []nvidiacomv1beta2.Backend{"vllm", "sglang", "trtllm"},
		Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
			SKU: ptr.To("h200_sxm"), Budget: ptr.To(int32(8)),
		}},
		Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
			ISL: ptr.To(int32(1024)), OSL: ptr.To(int32(128)), Concurrency: ptr.To(int32(10)),
		}},
		Objective: &nvidiacomv1beta2.ObjectiveSpec{
			Mode:   nvidiacomv1beta2.ObjectiveModeOptimize,
			Metric: ptr.To(nvidiacomv1beta2.ObjectiveMetric("Throughput")),
		},
		AISimulate:     &nvidiacomv1beta2.AISimulateSpec{Image: "planner:test"},
		Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestStructuralDefaultsV1beta2(t *testing.T) {
	tests := []struct {
		name  string
		input nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
		want  func() nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	}{
		{
			name:  "an empty spec receives every default",
			input: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{},
			want:  structuralDefaultedSpec,
		},
		{
			name: "empty parents receive their own defaults",
			input: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
				Workload:       &nvidiacomv1beta2.WorkloadSpec{},
				Objective:      &nvidiacomv1beta2.ObjectiveSpec{},
				Recommendation: &nvidiacomv1beta2.RecommendationSpec{},
			},
			want: structuralDefaultedSpec,
		},
		{
			name: "workload.static: {} is an opt-out and stays empty",
			input: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
				Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{}},
			},
			want: func() nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
				spec := structuralDefaultedSpec()
				spec.Workload = &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{}}
				return spec
			},
		},
		{
			name: "explicit values are kept",
			input: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
				Backends: []nvidiacomv1beta2.Backend{"vllm"},
				Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
					SKU: ptr.To("h100_sxm"), Budget: ptr.To(int32(4)),
				}},
				Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
					ISL: ptr.To(int32(10)),
				}},
				Objective: &nvidiacomv1beta2.ObjectiveSpec{
					Mode:   nvidiacomv1beta2.ObjectiveModeOptimize,
					Metric: ptr.To(nvidiacomv1beta2.ObjectiveMetric("TTFT")),
				},
				AISimulate:     &nvidiacomv1beta2.AISimulateSpec{Image: "mine:1"},
				Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 3},
			},
			want: func() nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
				return nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
					Backends: []nvidiacomv1beta2.Backend{"vllm"},
					Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
						SKU: ptr.To("h100_sxm"), Budget: ptr.To(int32(4)),
					}},
					Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
						ISL: ptr.To(int32(10)),
					}},
					Objective: &nvidiacomv1beta2.ObjectiveSpec{
						Mode:   nvidiacomv1beta2.ObjectiveModeOptimize,
						Metric: ptr.To(nvidiacomv1beta2.ObjectiveMetric("TTFT")),
					},
					AISimulate:     &nvidiacomv1beta2.AISimulateSpec{Image: "mine:1"},
					Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 3},
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := tt.input
			defaultDynamoGraphDeploymentRequestSpecV1beta2(&spec, structuralTestOptions())
			want := tt.want()
			if got, wantJSON := mustJSON(t, &spec), mustJSON(t, &want); got != wantJSON {
				t.Fatalf("defaulted spec mismatch\n got: %s\nwant: %s", got, wantJSON)
			}

			// Defaulting is idempotent.
			again := spec
			defaultDynamoGraphDeploymentRequestSpecV1beta2(&again, structuralTestOptions())
			if got, wantJSON := mustJSON(t, &again), mustJSON(t, &spec); got != wantJSON {
				t.Fatalf("second pass changed the spec\n got: %s\nwant: %s", got, wantJSON)
			}
		})
	}
}

func TestStructuralDefaultsV1beta2WithoutEnvironment(t *testing.T) {
	var spec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	defaultDynamoGraphDeploymentRequestSpecV1beta2(&spec, dynamoGraphDeploymentRequestV1beta2DefaultOptions{})
	if spec.Hardware != nil || spec.AISimulate != nil {
		t.Fatalf("hardware and aisimulate must stay absent without an environment: %s", mustJSON(t, &spec))
	}
	if spec.Workload == nil || spec.Objective == nil || spec.Recommendation == nil || spec.Backends == nil {
		t.Fatalf("static defaults must still apply: %s", mustJSON(t, &spec))
	}
}
