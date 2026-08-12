/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"context"
	"slices"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestDynamoGraphDeploymentRequestDeprecatedOverrideNameWarnings(t *testing.T) {
	tests := []struct {
		name         string
		overrideJSON string
		wantWarnings []string
	}{
		{
			name: "v1beta1 component names",
			overrideJSON: `{
				"apiVersion":"nvidia.com/v1beta1",
				"kind":"DynamoGraphDeployment",
				"spec":{"components":[
					{"name":"VllmWorker"},
					{"name":"TRTLLMPrefillWorker"},
					{"name":"SGLangDecodeWorker"}
				]}
			}`,
			wantWarnings: []string{
				`spec.overrides.dgd.spec.components[name=SGLangDecodeWorker]: generated component name "SGLangDecodeWorker" is deprecated; use worker for aggregate deployments or decode for disaggregated deployments`,
				`spec.overrides.dgd.spec.components[name=TRTLLMPrefillWorker]: generated component name "TRTLLMPrefillWorker" is deprecated; use prefill`,
				`spec.overrides.dgd.spec.components[name=VllmWorker]: generated component name "VllmWorker" is deprecated; use worker`,
			},
		},
		{
			name: "v1alpha1 service names",
			overrideJSON: `{
				"apiVersion":"nvidia.com/v1alpha1",
				"kind":"DynamoGraphDeployment",
				"spec":{"services":{"VllmDecodeWorker":{},"TRTLLMDecodeWorker":{}}}
			}`,
			wantWarnings: []string{
				`spec.overrides.dgd.spec.services.TRTLLMDecodeWorker: generated component name "TRTLLMDecodeWorker" is deprecated; use decode`,
				`spec.overrides.dgd.spec.services.VllmDecodeWorker: generated component name "VllmDecodeWorker" is deprecated; use worker for aggregate deployments or decode for disaggregated deployments`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := dgdrWithDGDOverride(tt.overrideJSON)
			warnings, err := NewDynamoGraphDeploymentRequestValidator().Validate(
				features.WithGate(context.Background(), features.Gates{GPUDiscovery: true}),
				request,
			)
			if err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
			if !slices.Equal(warnings, tt.wantWarnings) {
				t.Fatalf("Validate() warnings = %v, want %v", warnings, tt.wantWarnings)
			}
		})
	}
}

func TestDynamoGraphDeploymentRequestDeprecatedOverrideNameWarningOnUpdate(t *testing.T) {
	oldRequest := dgdrWithDGDOverride("")
	newRequest := dgdrWithDGDOverride(`{
		"apiVersion":"nvidia.com/v1beta1",
		"kind":"DynamoGraphDeployment",
		"spec":{"components":[{"name":"VllmPrefillWorker"}]}
	}`)

	warnings, err := NewDynamoGraphDeploymentRequestValidator().ValidateUpdate(
		features.WithGate(context.Background(), features.Gates{GPUDiscovery: true}),
		oldRequest,
		newRequest,
	)
	if err != nil {
		t.Fatalf("ValidateUpdate() error = %v", err)
	}
	wantWarnings := []string{
		`spec.overrides.dgd.spec.components[name=VllmPrefillWorker]: generated component name "VllmPrefillWorker" is deprecated; use prefill`,
	}
	if !slices.Equal(warnings, wantWarnings) {
		t.Fatalf("ValidateUpdate() warnings = %v, want %v", warnings, wantWarnings)
	}
}

func dgdrWithDGDOverride(overrideJSON string) *nvidiacomv1beta1.DynamoGraphDeploymentRequest {
	request := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgdr", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Model:                  "Qwen/Qwen3-0.6B",
			Backend:                nvidiacomv1beta1.BackendTypeVllm,
			Image:                  "profiler:latest",
			RuntimeVersionOverride: "1.1.0",
			SearchStrategy:         nvidiacomv1beta1.SearchStrategyRapid,
		},
	}
	if overrideJSON != "" {
		request.Spec.Overrides = &nvidiacomv1beta1.OverridesSpec{
			DGD: &runtime.RawExtension{Raw: []byte(overrideJSON)},
		}
	}
	return request
}
