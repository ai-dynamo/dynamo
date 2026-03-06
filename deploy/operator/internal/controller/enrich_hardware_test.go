/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"context"
	"fmt"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	gpupkg "github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
)

func newFakeReconciler(nodes ...*corev1.Node) *DynamoGraphDeploymentRequestReconciler {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	objs := make([]client.Object, len(nodes))
	for i, n := range nodes {
		objs[i] = n
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
	return &DynamoGraphDeploymentRequestReconciler{
		Client:    fakeClient,
		APIReader: fakeClient,
		Recorder:  &record.FakeRecorder{},
	}
}

func gpuNode(name, product string, gpuCount int, vramMiB int) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				gpupkg.LabelGPUCount:   intStr(gpuCount),
				gpupkg.LabelGPUProduct: product,
				gpupkg.LabelGPUMemory:  intStr(vramMiB),
			},
		},
	}
}

func intStr(n int) string {
	return fmt.Sprintf("%d", n)
}

// TestEnrichHardwareFromDiscovery_UsesAICSystemIdentifier is the regression test for the
// bug where GPUSKU was set to the raw GFD product name (e.g. "NVIDIA-B200") instead of
// the AIC system identifier (e.g. "b200_sxm"), causing AIC support checks to always fail
// and forcing every model/backend to fall back to naive config generation.
func TestEnrichHardwareFromDiscovery_UsesAICSystemIdentifier(t *testing.T) {
	tests := []struct {
		name           string
		gfdProduct     string // raw GFD label value
		expectedGPUSKU string // what the profiler needs
	}{
		{
			name:           "B200 GFD label maps to AIC system identifier",
			gfdProduct:     "NVIDIA-B200",
			expectedGPUSKU: "b200_sxm",
		},
		{
			name:           "H200 GFD label maps to AIC system identifier",
			gfdProduct:     "NVIDIA-H200-SXM5-141GB",
			expectedGPUSKU: "h200_sxm",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := newFakeReconciler(gpuNode("gpu-node-1", tt.gfdProduct, 8, 141312))
			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}

			err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
			require.NoError(t, err)
			require.NotNil(t, dgdr.Spec.Hardware)
			assert.Equal(t, tt.expectedGPUSKU, string(dgdr.Spec.Hardware.GPUSKU),
				"GPUSKU should be the AIC system identifier, not the raw GFD product name %q", tt.gfdProduct)
		})
	}
}

// TestEnrichHardwareFromDiscovery_FallsBackToModelForUnknownGPU verifies that for GPUs
// not in the AIC support matrix, the raw GFD product name is used as a fallback.
func TestEnrichHardwareFromDiscovery_FallsBackToModelForUnknownGPU(t *testing.T) {
	r := newFakeReconciler(gpuNode("gpu-node-1", "Tesla-V100-SXM2-16GB", 8, 16384))
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}

	err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.NoError(t, err)
	require.NotNil(t, dgdr.Spec.Hardware)
	assert.Equal(t, "Tesla-V100-SXM2-16GB", string(dgdr.Spec.Hardware.GPUSKU),
		"Unknown GPU should fall back to raw model name")
}

// TestEnrichHardwareFromDiscovery_RejectsUnrecognizedGPUSKU verifies that a user-provided
// gpuSku that InferHardwareSystem cannot map to any AIC identifier is rejected with an error.
func TestEnrichHardwareFromDiscovery_RejectsUnrecognizedGPUSKU(t *testing.T) {
	vram := float64(16384)
	n := int32(8)
	r := newFakeReconciler()
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU:         "my-custom-gpu-xyz",
				VRAMMB:         &vram,
				NumGPUsPerNode: &n,
			},
		},
	}

	err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "my-custom-gpu-xyz")
}

// TestEnrichHardwareFromDiscovery_NormalizesUserProvidedGPUSKU verifies that a raw product
// name manually set by the user in the DGDR spec is normalized to AIC format.
func TestEnrichHardwareFromDiscovery_NormalizesUserProvidedGPUSKU(t *testing.T) {
	vram := float64(144384)
	n := int32(8)
	r := newFakeReconciler() // no nodes needed — all three fields pre-set, skips discovery
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU:         "NVIDIA-B200", // raw product name, should be normalized
				VRAMMB:         &vram,
				NumGPUsPerNode: &n,
			},
		},
	}

	err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.NoError(t, err)
	assert.Equal(t, "b200_sxm", string(dgdr.Spec.Hardware.GPUSKU),
		"User-provided raw product name should be normalized to AIC system identifier")
}
