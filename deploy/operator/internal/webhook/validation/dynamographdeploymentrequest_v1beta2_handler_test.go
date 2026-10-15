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
	"context"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
)

func validV1Beta2Request() *nvidiacomv1beta2.DynamoGraphDeploymentRequest {
	metric := nvidiacomv1beta2.ObjectiveMetric("ThroughputPerGPU")
	request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{}
	request.Name = "r"
	request.Namespace = "ns"
	request.Spec = nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    nvidiacomv1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM},
		Hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: ptr.To("h200_sxm"), Budget: ptr.To(int32(8))}},
		Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
			ISL: ptr.To(int32(4000)), OSL: ptr.To(int32(1000)), KVLoadRatio: nvidiacomv1beta2.KVLoadRatio{0, 1},
		}},
		Objective: &nvidiacomv1beta2.ObjectiveSpec{
			Mode:   nvidiacomv1beta2.ObjectiveModeOptimize,
			Metric: &metric,
			SLA:    &nvidiacomv1beta2.ObjectiveSLASpec{TTFTMs: ptr.To(2000.0), ITLMs: ptr.To(30.0)},
		},
		Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
	}
	return request
}

func TestV1Beta2ValidatorCreate(t *testing.T) {
	validator := NewDynamoGraphDeploymentRequestV1Beta2Validator()
	if _, err := validator.Validate(context.Background(), validV1Beta2Request()); err != nil {
		t.Fatalf("valid request rejected: %v", err)
	}

	missing := validV1Beta2Request()
	missing.Spec.Hardware = nil
	if _, err := validator.Validate(context.Background(), missing); !errors.IsInvalid(err) {
		t.Fatalf("unresolved hardware must be Invalid, got %v", err)
	}
}

func TestV1Beta2ValidatorUpdate(t *testing.T) {
	validator := NewDynamoGraphDeploymentRequestV1Beta2Validator()
	oldRequest := validV1Beta2Request()
	oldRequest.Spec.Hardware = nil // admitted earlier with an unresolved violation

	// A status or metadata write leaves the spec alone and is admitted.
	same := oldRequest.DeepCopy()
	same.Annotations = map[string]string{"k": "v"}
	if _, err := validator.ValidateUpdate(context.Background(), oldRequest, same); err != nil {
		t.Fatalf("spec-neutral update rejected: %v", err)
	}

	// An unrelated spec edit does not trip over the preexisting violation.
	edited := oldRequest.DeepCopy()
	edited.Spec.Recommendation.MaxCandidates = 3
	if _, err := validator.ValidateUpdate(context.Background(), oldRequest, edited); err != nil {
		t.Fatalf("preexisting violation blocked an unrelated edit: %v", err)
	}

	// An edit that introduces a new violation is rejected.
	broken := validV1Beta2Request()
	broken.Spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{
		Raw: []byte(`{"workload":{"concurrency":4}}`),
	}}
	if _, err := validator.ValidateUpdate(context.Background(), validV1Beta2Request(), broken); !errors.IsInvalid(err) {
		t.Fatalf("introduced violation must be Invalid, got %v", err)
	}
}
