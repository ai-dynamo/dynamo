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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
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

// newFakeReconcilerWithDiscovery returns a reconciler wired up with a non-nil
// GPUDiscovery (so DiscoverGPUsFromDCGM can run against the fake client) and
// an OperatorConfiguration with GPU.DiscoveryEnabled=true so the node-label
// fallback branch is exercised.
//
// The embedded scraper is never invoked by these tests because the fake
// client does not carry DCGM exporter pods — DiscoverGPUsFromDCGM fails early
// and the reconciler falls back to gpu.DiscoverGPUs.
func newFakeReconcilerWithDiscovery(nodes ...*corev1.Node) *DynamoGraphDeploymentRequestReconciler {
	r := newFakeReconciler(nodes...)
	r.GPUDiscovery = gpupkg.NewGPUDiscovery(gpupkg.ScrapeMetricsEndpoint)
	r.GPUDiscoveryCache = gpupkg.NewGPUDiscoveryCache()
	r.Config = &configv1alpha1.OperatorConfiguration{
		GPU: configv1alpha1.GPUConfiguration{
			DiscoveryEnabled: ptr.To(true),
		},
	}
	return r
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
		gfdProduct     string                      // raw GFD label value
		expectedGPUSKU nvidiacomv1beta1.GPUSKUType // what the profiler needs
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
			vram := float64(141312)
			gpus := int32(8)

			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Hardware: &nvidiacomv1beta1.HardwareSpec{
						GPUSKU:         tt.expectedGPUSKU,
						VRAMMB:         &vram,
						NumGPUsPerNode: &gpus,
					},
				},
			}
			err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)

			require.NoError(t, err)
			require.NotNil(t, dgdr.Spec.Hardware)
			assert.Equal(t, string(tt.expectedGPUSKU), string(dgdr.Spec.Hardware.GPUSKU),
				"GPUSKU should be the AIC system identifier, not the raw GFD product name %q", tt.gfdProduct)
		})
	}
}

// TestEnrichHardwareFromDiscovery_FallsBackToModelForUnknownGPU verifies that for GPUs
// not in the AIC support matrix, the raw GFD product name is used as a fallback.
func TestEnrichHardwareFromDiscovery_FallsBackToModelForUnknownGPU(t *testing.T) {
	r := newFakeReconciler(gpuNode("gpu-node-1", "Tesla-V100-SXM2-16GB", 8, 16384))
	vram := float64(16384)
	gpus := int32(8)

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU:         "Tesla-V100-SXM2-16GB",
				VRAMMB:         &vram,
				NumGPUsPerNode: &gpus,
			},
		},
	}

	err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.NoError(t, err)
	require.NotNil(t, dgdr.Spec.Hardware)
	assert.Equal(t, "Tesla-V100-SXM2-16GB", string(dgdr.Spec.Hardware.GPUSKU),
		"Unknown GPU should fall back to raw model name")
}

// TestEnrichHardwareFromDiscovery_PartialSpecNoPanic is the regression test for
// defect 2: a DGDR spec with a partial spec.hardware (for example only
// numGpusPerNode set) used to be treated as "fully specified", skip discovery,
// and then dereference a nil gpuInfo in the enrichment block, crashing the
// reconciler with a nil pointer panic.
//
// The fix runs discovery whenever any required field is missing, so partial
// specs now successfully enrich (via DCGM — or, as in this test, the
// node-label fallback when DCGM is unreachable).
func TestEnrichHardwareFromDiscovery_PartialSpecNoPanic(t *testing.T) {
	r := newFakeReconcilerWithDiscovery(gpuNode("gpu-node-1", "NVIDIA-H200-SXM5-141GB", 8, 143771))
	gpus := int32(8)

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				NumGPUsPerNode: &gpus,
			},
		},
	}

	require.NotPanics(t, func() {
		err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
		require.NoError(t, err)
	}, "partial spec.hardware must not trigger a nil pointer dereference")

	require.NotNil(t, dgdr.Spec.Hardware)
	assert.Equal(t, "h200_sxm", string(dgdr.Spec.Hardware.GPUSKU),
		"missing GPUSKU should be filled from node-label fallback")
	require.NotNil(t, dgdr.Spec.Hardware.VRAMMB)
	assert.InDelta(t, 143771, *dgdr.Spec.Hardware.VRAMMB, 0.001,
		"missing VRAMMB should be filled from node-label fallback")
	require.NotNil(t, dgdr.Spec.Hardware.NumGPUsPerNode)
	assert.Equal(t, int32(8), *dgdr.Spec.Hardware.NumGPUsPerNode,
		"user-provided NumGPUsPerNode must be preserved")
}

// TestEnrichHardwareFromDiscovery_NodeLabelFallback is the regression test for
// defect 2's missing-fallback symptom: in environments where DCGM exporter
// pods cannot be scraped (e.g. vCluster) but GFD node labels are present, the
// enrichment path used to return the DCGM failure error without attempting
// node-label discovery, leaving the DGDR stuck.
func TestEnrichHardwareFromDiscovery_NodeLabelFallback(t *testing.T) {
	r := newFakeReconcilerWithDiscovery(gpuNode("gpu-node-1", "NVIDIA-H200-SXM5-141GB", 8, 143771))

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{},
	}

	err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.NoError(t, err, "node-label fallback should succeed when DCGM is unreachable")
	require.NotNil(t, dgdr.Spec.Hardware)
	assert.Equal(t, "h200_sxm", string(dgdr.Spec.Hardware.GPUSKU))
	require.NotNil(t, dgdr.Spec.Hardware.NumGPUsPerNode)
	assert.Equal(t, int32(8), *dgdr.Spec.Hardware.NumGPUsPerNode)
	require.NotNil(t, dgdr.Spec.Hardware.VRAMMB)
	assert.InDelta(t, 143771, *dgdr.Spec.Hardware.VRAMMB, 0.001)
}

// TestEnrichHardwareFromDiscovery_NodeLabelFallbackDisabled verifies that when
// node read access is explicitly disabled (helm gpuDiscovery.enabled=false),
// the fallback branch is skipped and the DCGM error is returned verbatim so
// the user sees the real failure reason rather than a misleading one.
func TestEnrichHardwareFromDiscovery_NodeLabelFallbackDisabled(t *testing.T) {
	r := newFakeReconcilerWithDiscovery(gpuNode("gpu-node-1", "NVIDIA-H200", 8, 143771))
	r.Config.GPU.DiscoveryEnabled = ptr.To(false)

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{},
	}

	err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.Error(t, err, "with fallback disabled, DCGM failure must surface")
}

// TestValidateGPUHardwareInfo_NamespaceScopedAttemptsDiscovery is the
// regression test for defect 1: namespace-scoped installs used to
// short-circuit with ValidationFailed before attempting any discovery, even
// when DCGM was reachable and/or GFD labels were populated. The fix drops
// the namespace-scope short-circuit and runs discovery uniformly.
func TestValidateGPUHardwareInfo_NamespaceScopedAttemptsDiscovery(t *testing.T) {
	r := newFakeReconcilerWithDiscovery(gpuNode("gpu-node-1", "NVIDIA-H200", 8, 143771))
	// Simulate namespace-scoped operator install. The production controller
	// still reads this deprecated field (dynamographdeploymentrequest_controller.go
	// and dynamographdeployment_controller.go both gate RBAC setup on it), so the
	// regression test must exercise it directly.
	r.Config.Namespace.Restricted = "djangoz" //nolint:staticcheck // SA1019: testing deprecated namespace-restricted mode the controller still supports

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{},
	}

	err := r.validateGPUHardwareInfo(context.Background(), dgdr)
	assert.NoError(t, err,
		"namespace-scoped install must not short-circuit discovery when node labels are available")
}
