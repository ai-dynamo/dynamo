/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func newConversionScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := AddToScheme(scheme); err != nil {
		t.Fatalf("register v1alpha1 scheme: %v", err)
	}
	if err := v1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("register v1beta1 scheme: %v", err)
	}
	return scheme
}

func TestGeneratedSchemeConversionsUseCustomConverters(t *testing.T) {
	scheme := newConversionScheme(t)
	pvcName := "model-cache"
	enableDiscovery := false

	tests := []struct {
		name       string
		annotation string
		src        runtime.Object
		hub        runtime.Object
		got        runtime.Object
	}{
		{
			name:       "DCD",
			annotation: annDCDStatus,
			src: &DynamoComponentDeployment{
				Status: DynamoComponentDeploymentStatus{PodSelector: map[string]string{"component": "decode"}},
			},
			hub: &v1beta1.DynamoComponentDeployment{},
			got: &DynamoComponentDeployment{},
		},
		{
			name:       "DGD",
			annotation: annDGDSpec,
			src: &DynamoGraphDeployment{
				Spec: DynamoGraphDeploymentSpec{PVCs: []PVC{{Name: &pvcName}}},
			},
			hub: &v1beta1.DynamoGraphDeployment{},
			got: &DynamoGraphDeployment{},
		},
		{
			name:       "DGDR",
			annotation: annDGDRSpec,
			src: &DynamoGraphDeploymentRequest{
				Spec:   DynamoGraphDeploymentRequestSpec{EnableGPUDiscovery: &enableDiscovery},
				Status: DynamoGraphDeploymentRequestStatus{State: DGDRStatePending},
			},
			hub: &v1beta1.DynamoGraphDeploymentRequest{},
			got: &DynamoGraphDeploymentRequest{},
		},
	}

	for i := range tests {
		tt := &tests[i]
		t.Run(tt.name, func(t *testing.T) {
			if err := scheme.Convert(tt.src, tt.hub, nil); err != nil {
				t.Fatalf("alpha to beta: %v", err)
			}
			annotations := tt.hub.(metav1.Object).GetAnnotations()
			if _, ok := annotations[tt.annotation]; !ok {
				t.Fatalf("alpha-only data was not persisted in %q", tt.annotation)
			}

			if err := scheme.Convert(tt.hub, tt.got, nil); err != nil {
				t.Fatalf("beta to alpha: %v", err)
			}
			if diff := cmp.Diff(tt.src, tt.got); diff != "" {
				t.Fatalf("scheme round-trip mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
