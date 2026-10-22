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

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/runtime"
)

func runWithParameters(raw string) *v1beta2.DynamoGraphDeploymentRun {
	run := &v1beta2.DynamoGraphDeploymentRun{}
	run.Spec.AISimulate = &v1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{Raw: []byte(raw)}}
	return run
}

func TestInputForRunReturnsTheFrozenObjectUnchanged(t *testing.T) {
	input, err := InputForRun(runWithParameters(`{"search_space":{"gpu_budget":4},"workload":{"osl":128}}`))
	if err != nil {
		t.Fatalf("InputForRun: %v", err)
	}
	workload, _ := input["workload"].(map[string]any)
	if _, found := workload["isl"]; found || workload["osl"] != float64(128) {
		t.Fatalf("workload = %v, want exactly the stored keys", workload)
	}
}

func TestInputForRunRejectsAMissingOrMalformedInput(t *testing.T) {
	for name, run := range map[string]*v1beta2.DynamoGraphDeploymentRun{
		"no aisimulate": {},
		"no parameters": {Spec: v1beta2.DynamoGraphDeploymentRunSpec{
			NativeSearchInput: v1beta2.NativeSearchInput{AISimulate: &v1beta2.AISimulateSpec{}},
		}},
		"empty object":  runWithParameters(`{}`),
		"not an object": runWithParameters(`[1]`),
		"not json":      runWithParameters(`{`),
	} {
		if _, err := InputForRun(run); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}
