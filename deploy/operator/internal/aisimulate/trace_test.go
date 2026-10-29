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

package aisimulate

import (
	"testing"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
)

func withTrace(spec *v1beta2.DynamoGraphDeploymentRequestSpec, speedup *float64) *v1beta2.DynamoGraphDeploymentRequestSpec {
	spec.Workload = &v1beta2.WorkloadSpec{Trace: &v1beta2.TraceWorkloadSpec{
		Format: v1beta2.TraceFormatDynamo,
		Source: v1beta2.TraceSourceSpec{PVC: &v1beta2.TracePVCSource{ClaimName: "request-traces", Path: "retune/w-1.jsonl.gz"}},
	}}
	if speedup != nil {
		spec.Workload.Trace.Replay = &v1beta2.TraceReplaySpec{ArrivalSpeedupRatio: speedup}
	}
	return spec
}

func TestLowerTraceWorkload(t *testing.T) {
	tests := []struct {
		name        string
		spec        *v1beta2.DynamoGraphDeploymentRequestSpec
		wantTraffic string
	}{
		{
			name: "recorded arrival times without a request stop",
			spec: withTrace(defaultedSpec(), nil),
			wantTraffic: `{"load":{"type":"trace_timestamps"},` +
				`"source":{"format":"dynamo","paths":["/var/run/aisimulate/trace/trace.jsonl.gz"],"type":"trace"}}`,
		},
		{
			name: "an arrival speedup lowers to traffic.load.speedup",
			spec: withTrace(defaultedSpec(), new(2.5)),
			wantTraffic: `{"load":{"speedup":2.5,"type":"trace_timestamps"},` +
				`"source":{"format":"dynamo","paths":["/var/run/aisimulate/trace/trace.jsonl.gz"],"type":"trace"}}`,
		},
		{
			name: "a stop override still applies",
			spec: withInputOverrides(withTrace(defaultedSpec(), nil), `{"traffic":{"stop":{"max_virtual_time_seconds":600}}}`),
			wantTraffic: `{"load":{"type":"trace_timestamps"},` +
				`"source":{"format":"dynamo","paths":["/var/run/aisimulate/trace/trace.jsonl.gz"],"type":"trace"},` +
				`"stop":{"max_virtual_time_seconds":600}}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			traffic := nested(t, lower(t, tt.spec), "traffic").(map[string]any)
			if got := encode(t, traffic); got != tt.wantTraffic {
				t.Fatalf("traffic\n got: %s\nwant: %s", got, tt.wantTraffic)
			}
		})
	}
}

func TestTracePathIsOneFixedFileUnderTheMount(t *testing.T) {
	for claimPath, want := range map[string]string{
		"retune/w-1.jsonl.gz": "/var/run/aisimulate/trace/trace.jsonl.gz",
		"retune/w-1.jsonl":    "/var/run/aisimulate/trace/trace.jsonl",
		"../../etc/passwd":    "/var/run/aisimulate/trace/trace.jsonl",
	} {
		if got := TracePath(claimPath); got != want {
			t.Errorf("TracePath(%q) = %q, want %q", claimPath, got, want)
		}
	}
}
