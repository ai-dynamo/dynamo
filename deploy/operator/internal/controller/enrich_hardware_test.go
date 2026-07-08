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
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	gpupkg "github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"k8s.io/utils/ptr"
)

func newFakeReconciler(objs ...client.Object) *DynamoGraphDeploymentRequestReconciler {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
	return &DynamoGraphDeploymentRequestReconciler{
		Client:    fakeClient,
		APIReader: fakeClient,
		Recorder:  &record.FakeRecorder{},
		Config:    &configv1alpha1.OperatorConfiguration{},
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

func dcgmPod(name, ip string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "gpu-operator",
			Labels: map[string]string{gpupkg.LabelApp: gpupkg.LabelValueNvidiaDCGMExporter}},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: ip},
	}
}

func xpumdPod(name, ip string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "intel-xpum",
			Labels: map[string]string{gpupkg.LabelAppKubernetesName: gpupkg.LabelValueXPUMD}},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: ip},
	}
}

func TestGPUDiscoveryEnabledDefaults(t *testing.T) {
	assert.True(t, (*DynamoGraphDeploymentRequestReconciler)(nil).gpuDiscoveryEnabled())
	assert.True(t, (&DynamoGraphDeploymentRequestReconciler{}).gpuDiscoveryEnabled())
	assert.True(t, (&DynamoGraphDeploymentRequestReconciler{
		Config: &configv1alpha1.OperatorConfiguration{},
	}).gpuDiscoveryEnabled())
	assert.False(t, (&DynamoGraphDeploymentRequestReconciler{
		Config: &configv1alpha1.OperatorConfiguration{
			GPU: configv1alpha1.GPUConfiguration{
				DiscoveryEnabled: ptr.To(false),
			},
		},
	}).gpuDiscoveryEnabled())
}

func TestEnrichHardwareFromDiscovery_SkipsOptionalMetadataWithoutAPIReader(t *testing.T) {
	r := &DynamoGraphDeploymentRequestReconciler{
		Config: &configv1alpha1.OperatorConfiguration{},
	}
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU:         nvidiacomv1beta1.GPUSKUTypeH100SXM,
				VRAMMB:         ptr.To(81920.0),
				NumGPUsPerNode: ptr.To(int32(8)),
				TotalGPUs:      ptr.To(int32(16)),
			},
		},
	}

	changed, err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.NoError(t, err)
	assert.False(t, changed)
	assert.Empty(t, dgdr.Spec.Hardware.Interconnect)
	assert.Nil(t, dgdr.Spec.Hardware.RDMA)
}

func TestEnrichHardwareFromDiscovery_RequiredFieldsMissingWithoutAPIReaderFails(t *testing.T) {
	r := &DynamoGraphDeploymentRequestReconciler{
		Config: &configv1alpha1.OperatorConfiguration{},
	}
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{},
		},
	}

	changed, err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "APIReader is not configured")
	assert.False(t, changed)
}

func TestEnrichHardwareFromDiscovery_SkipsOptionalMetadataWhenNodeDiscoveryDisabled(t *testing.T) {
	node := gpuNode("gpu-node-rdma", "H100-SXM5-80GB", 8, 81920)
	node.Labels[gpupkg.LabelNFDRDMAAvailable] = "true"
	r := newFakeReconciler(node)
	r.GPUDiscovery = nil
	r.Config.GPU.DiscoveryEnabled = ptr.To(false)

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU:         nvidiacomv1beta1.GPUSKUTypeH100SXM,
				VRAMMB:         ptr.To(81920.0),
				NumGPUsPerNode: ptr.To(int32(8)),
				TotalGPUs:      ptr.To(int32(16)),
			},
		},
	}

	changed, err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.NoError(t, err)
	assert.False(t, changed)
	assert.Nil(t, dgdr.Spec.Hardware.RDMA)
}

func TestEnrichHardwareFromDiscovery(t *testing.T) {
	tests := []struct {
		name string

		// Input hardware spec (nil fields = not set by user).
		hardware *nvidiacomv1beta1.HardwareSpec

		// Discovery mock: what DCGM returns. Nil means no discovery available.
		discoveredGPU *gpupkg.GPUInfo

		// Expected outcome.
		wantErr       string // non-empty means error expected, substring match
		wantGPUSKU    string
		wantVRAM      float64
		wantGPUsNode  int32
		wantTotalGPUs int32
		wantChanged   bool
	}{
		{
			name: "required fields set, no discovery available",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU: "h100_sxm", VRAMMB: ptr.To(81920.0),
				NumGPUsPerNode: ptr.To(int32(8)), TotalGPUs: ptr.To(int32(16)),
			},
			wantGPUSKU: "h100_sxm", wantVRAM: 81920, wantGPUsNode: 8, wantTotalGPUs: 16, wantChanged: false,
		},
		{
			name:          "nothing set, full discovery",
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "H100-SXM5-80GB", VRAMPerGPU: 81920},
			wantGPUSKU:    "h100_sxm", wantVRAM: 81920, wantGPUsNode: 8, wantTotalGPUs: 8, wantChanged: true,
		},
		{
			name:          "nothing set, V100 discovered",
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "Tesla-V100-SXM2-16GB", VRAMPerGPU: 16384},
			wantGPUSKU:    "v100_sxm", wantVRAM: 16384, wantGPUsNode: 8, wantTotalGPUs: 8, wantChanged: true,
		},
		{
			name:          "nothing set, unknown GPU falls back to model name",
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 4, Model: "FutureGPU-X1000", VRAMPerGPU: 65536},
			wantGPUSKU:    "FutureGPU-X1000", wantVRAM: 65536, wantGPUsNode: 4, wantTotalGPUs: 4, wantChanged: true,
		},
		{
			name: "only totalGpus missing, discovery fills it",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU: "b200_sxm", VRAMMB: ptr.To(141312.0), NumGPUsPerNode: ptr.To(int32(8)),
			},
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "B200-SXM-180GB", VRAMPerGPU: 141312},
			wantGPUSKU:    "b200_sxm", wantVRAM: 141312, wantGPUsNode: 8, wantTotalGPUs: 8, wantChanged: true,
		},
		{
			name: "only gpuSku missing, discovery fills it",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				VRAMMB: ptr.To(81920.0), NumGPUsPerNode: ptr.To(int32(8)), TotalGPUs: ptr.To(int32(16)),
			},
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "H200-SXM5-141GB", VRAMPerGPU: 141312},
			wantGPUSKU:    "h200_sxm", wantVRAM: 81920, wantGPUsNode: 8, wantTotalGPUs: 16, wantChanged: true, // user overrides win
		},
		{
			name: "vramMb and numGpusPerNode override discovery",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU: "a100_sxm", VRAMMB: ptr.To(40960.0), NumGPUsPerNode: ptr.To(int32(4)),
			},
			discoveredGPU: &gpupkg.GPUInfo{NodeName: "n1", GPUsPerNode: 8, Model: "A100-SXM4-80GB", VRAMPerGPU: 81920},
			wantGPUSKU:    "a100_sxm", wantVRAM: 40960, wantGPUsNode: 4, wantTotalGPUs: 8, wantChanged: true,
		},
		{
			name:    "no fields set, discovery fails",
			wantErr: "auto-discovery failed",
		},
		{
			name: "three fields set, discovery fails",
			hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU: "h100_sxm", VRAMMB: ptr.To(81920.0), NumGPUsPerNode: ptr.To(int32(8)),
			},
			wantErr: "auto-discovery failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var r *DynamoGraphDeploymentRequestReconciler
			if tt.discoveredGPU != nil {
				// Set up a DCGM pod and mock scraper so discovery works.
				r = newFakeReconciler(dcgmPod("dcgm-exporter", "10.0.0.1"))
				r.GPUDiscovery = gpupkg.NewGPUDiscovery(func(ctx context.Context, endpoint string) (*gpupkg.GPUInfo, error) {
					return tt.discoveredGPU, nil
				})
				r.GPUDiscoveryCache = gpupkg.NewGPUDiscoveryCache()
			} else if tt.wantErr != "" {
				// No discovery — will fail if discovery is attempted.
				r = newFakeReconciler()
			} else {
				// All fields set — discovery not needed, no mock required.
				r = newFakeReconciler()
			}

			hw := tt.hardware
			if hw == nil {
				hw = &nvidiacomv1beta1.HardwareSpec{}
			}
			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
					Hardware: hw,
				},
			}

			changed, err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantChanged, changed)
			require.NotNil(t, dgdr.Spec.Hardware)
			assert.Equal(t, tt.wantGPUSKU, string(dgdr.Spec.Hardware.GPUSKU))
			assert.Equal(t, tt.wantVRAM, *dgdr.Spec.Hardware.VRAMMB)
			assert.Equal(t, tt.wantGPUsNode, *dgdr.Spec.Hardware.NumGPUsPerNode)
			assert.Equal(t, tt.wantTotalGPUs, *dgdr.Spec.Hardware.TotalGPUs)
		})
	}
}

func TestEnrichHardwareFromDiscovery_B60UsesXPUMD(t *testing.T) {
	t.Setenv("INTEL_XPU_METRICS_ENDPOINT_TEMPLATE", "")

	r := newFakeReconciler(xpumdPod("xpumd", "10.0.0.2"))
	r.GPUDiscovery = gpupkg.NewGPUDiscoveryWithScrapers(
		func(ctx context.Context, endpoint string) (*gpupkg.GPUInfo, error) {
			t.Fatalf("DCGM scraper should not be called for explicit B60 discovery")
			return nil, nil
		},
		func(ctx context.Context, endpoint string) (*gpupkg.GPUInfo, error) {
			assert.Contains(t, endpoint, "10.0.0.2:8080")
			return &gpupkg.GPUInfo{
				NodeName:    "xpu-node",
				GPUsPerNode: 3,
				Model:       "Intel(R) Graphics [0xe211]",
				VRAMPerGPU:  24480,
				System:      "b60",
			}, nil
		},
	)
	r.GPUDiscoveryCache = gpupkg.NewGPUDiscoveryCache()

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU: "b60",
			},
		},
	}

	changed, err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)

	require.NoError(t, err)
	assert.True(t, changed)
	require.NotNil(t, dgdr.Spec.Hardware)
	assert.Equal(t, "b60", string(dgdr.Spec.Hardware.GPUSKU))
	assert.Equal(t, 24480.0, *dgdr.Spec.Hardware.VRAMMB)
	assert.Equal(t, int32(3), *dgdr.Spec.Hardware.NumGPUsPerNode)
	assert.Equal(t, int32(3), *dgdr.Spec.Hardware.TotalGPUs)
}

func TestEnrichHardwareFromDiscovery_WritesOptionalHardwareMetadata(t *testing.T) {
	r := newFakeReconciler()
	r.GPUDiscovery = gpupkg.NewGPUDiscovery(nil)
	r.GPUDiscoveryCache = gpupkg.NewGPUDiscoveryCache()
	r.GPUDiscoveryCache.Set("", &gpupkg.GPUInfo{
		NodeName:      "n1",
		GPUsPerNode:   8,
		NodesWithGPUs: 2,
		Model:         "NVIDIA H100 80GB HBM3",
		VRAMPerGPU:    81079,
		System:        nvidiacomv1beta1.GPUSKUTypeH100PCIe,
		Interconnect:  "pcie",
		RDMAEnabled:   true,
		RDMAType:      "rdma",
	}, time.Minute)

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{},
		},
	}

	changed, err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.NoError(t, err)
	require.True(t, changed)
	require.NotNil(t, dgdr.Spec.Hardware)
	assert.Equal(t, nvidiacomv1beta1.GPUSKUTypeH100PCIe, dgdr.Spec.Hardware.GPUSKU)
	assert.Equal(t, "pcie", dgdr.Spec.Hardware.Interconnect)
	require.NotNil(t, dgdr.Spec.Hardware.RDMA)
	assert.True(t, *dgdr.Spec.Hardware.RDMA)
}

func TestEnrichHardwareFromDiscovery_FillsOptionalMetadataWhenRequiredFieldsSet(t *testing.T) {
	r := newFakeReconciler()
	r.GPUDiscovery = gpupkg.NewGPUDiscovery(nil)
	r.GPUDiscoveryCache = gpupkg.NewGPUDiscoveryCache()
	r.GPUDiscoveryCache.Set(nvidiacomv1beta1.GPUSKUTypeH100PCIe, &gpupkg.GPUInfo{
		NodeName:      "n1",
		GPUsPerNode:   8,
		NodesWithGPUs: 4,
		Model:         "NVIDIA H100 80GB HBM3",
		VRAMPerGPU:    81079,
		System:        nvidiacomv1beta1.GPUSKUTypeH100PCIe,
		Interconnect:  "pcie",
		RDMAEnabled:   true,
		RDMAType:      "rdma",
	}, time.Minute)

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU:         nvidiacomv1beta1.GPUSKUTypeH100PCIe,
				VRAMMB:         ptr.To(81920.0),
				NumGPUsPerNode: ptr.To(int32(8)),
				TotalGPUs:      ptr.To(int32(16)),
			},
		},
	}

	changed, err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.NoError(t, err)
	require.True(t, changed)
	assert.Equal(t, nvidiacomv1beta1.GPUSKUTypeH100PCIe, dgdr.Spec.Hardware.GPUSKU)
	assert.Equal(t, float64(81920), *dgdr.Spec.Hardware.VRAMMB)
	assert.Equal(t, int32(8), *dgdr.Spec.Hardware.NumGPUsPerNode)
	assert.Equal(t, int32(16), *dgdr.Spec.Hardware.TotalGPUs)
	assert.Equal(t, "pcie", dgdr.Spec.Hardware.Interconnect)
	require.NotNil(t, dgdr.Spec.Hardware.RDMA)
	assert.True(t, *dgdr.Spec.Hardware.RDMA)
}

func TestCreateProfilingJobPersistsDiscoveredHardware(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, batchv1.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "persist-hardware",
			Namespace: "default",
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "test-model",
			Backend: nvidiacomv1beta1.BackendTypeVllm,
			Image:   "test-profiler:latest",
		},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgdr).Build()

	cache := gpupkg.NewGPUDiscoveryCache()
	cache.Set("", &gpupkg.GPUInfo{
		NodeName:      "n1",
		GPUsPerNode:   8,
		NodesWithGPUs: 2,
		Model:         "NVIDIA H100 80GB HBM3",
		VRAMPerGPU:    81079,
		System:        nvidiacomv1beta1.GPUSKUTypeH100PCIe,
		Interconnect:  "pcie",
		RDMAEnabled:   true,
		RDMAType:      "rdma",
	}, time.Minute)

	r := &DynamoGraphDeploymentRequestReconciler{
		Client:            fakeClient,
		APIReader:         fakeClient,
		Recorder:          &record.FakeRecorder{},
		Config:            &configv1alpha1.OperatorConfiguration{},
		GPUDiscovery:      gpupkg.NewGPUDiscovery(nil),
		GPUDiscoveryCache: cache,
		RBACManager:       &MockRBACManager{},
	}

	var fetched nvidiacomv1beta1.DynamoGraphDeploymentRequest
	require.NoError(t, fakeClient.Get(ctx, types.NamespacedName{Name: dgdr.Name, Namespace: dgdr.Namespace}, &fetched))

	requeue, err := r.createProfilingJob(ctx, &fetched)
	require.NoError(t, err)
	require.True(t, requeue)

	var stored nvidiacomv1beta1.DynamoGraphDeploymentRequest
	require.NoError(t, fakeClient.Get(ctx, types.NamespacedName{Name: dgdr.Name, Namespace: dgdr.Namespace}, &stored))
	require.NotNil(t, stored.Spec.Hardware)
	assert.Equal(t, nvidiacomv1beta1.GPUSKUTypeH100PCIe, stored.Spec.Hardware.GPUSKU)
	assert.Equal(t, float64(81079), *stored.Spec.Hardware.VRAMMB)
	assert.Equal(t, int32(8), *stored.Spec.Hardware.NumGPUsPerNode)
	assert.Equal(t, int32(16), *stored.Spec.Hardware.TotalGPUs)
	assert.Equal(t, "pcie", stored.Spec.Hardware.Interconnect)
	require.NotNil(t, stored.Spec.Hardware.RDMA)
	assert.True(t, *stored.Spec.Hardware.RDMA)

	requeue, err = r.createProfilingJob(ctx, &stored)
	require.NoError(t, err)
	require.True(t, requeue)

	requeue, err = r.createProfilingJob(ctx, &stored)
	require.NoError(t, err)
	require.False(t, requeue)

	job := &batchv1.Job{}
	require.NoError(t, fakeClient.Get(ctx, types.NamespacedName{
		Name:      getProfilingJobName(&stored),
		Namespace: stored.Namespace,
	}, job))
}

// TestGetGPUDiscoveryFailureReason_DCGMNotEnabledMatches is a regression guard:
// GetGPUDiscoveryFailureReason lowercases the error before matching, so matcher
// substrings must be lowercase. The "DCGM is not enabled in the GPU Operator"
// case previously used a mixed-case substring that could never match and fell
// through to "unknown".
func TestGetGPUDiscoveryFailureReason_DCGMNotEnabledMatches(t *testing.T) {
	err := fmt.Errorf("DCGM is not enabled in the GPU Operator (check GPU Operator configuration and permissions)")
	reason := GetGPUDiscoveryFailureReason(err)
	assert.NotEqual(t, "unknown", reason)
	assert.Equal(t, "DCGM is not enabled in the GPU Operator (check GPU Operator configuration and permissions)", reason)
}

// TestGetGPUDiscoveryFailureReason_IntelFailuresNotMislabeledAsDCGM verifies
// that aggregated Intel XPUMD discovery errors — which embed generic substrings
// like "http get", "list pods", "parse prometheus metrics", and
// "metrics endpoint ... status" — resolve to Intel-worded reasons rather than
// the DCGM-worded cases. This guards the ordering of GetGPUDiscoveryFailureReason.
func TestGetGPUDiscoveryFailureReason_IntelFailuresNotMislabeledAsDCGM(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "intel scrape failure wrapping HTTP GET",
			err:  fmt.Errorf("failed to scrape any Intel XPUMD exporter pod: %v", []error{fmt.Errorf("pod xpumd (10.0.0.2): %w", fmt.Errorf("HTTP GET %s failed: %w", "http://10.0.0.2:9090/metrics", fmt.Errorf("connection refused")))}),
			want: "failed to scrape any Intel XPUMD exporter pod (check XPUMD pod status and network connectivity)",
		},
		{
			name: "intel scrape failure wrapping non-200 status",
			err:  fmt.Errorf("failed to scrape any Intel XPUMD exporter pod: %v", []error{fmt.Errorf("pod xpumd (10.0.0.2): %w", fmt.Errorf("metrics endpoint %s returned status %d", "http://10.0.0.2:9090/metrics", 503))}),
			want: "failed to scrape any Intel XPUMD exporter pod (check XPUMD pod status and network connectivity)",
		},
		{
			name: "intel scrape failure wrapping parse error",
			err:  fmt.Errorf("failed to scrape any Intel XPUMD exporter pod: %v", []error{fmt.Errorf("pod xpumd (10.0.0.2): %w", fmt.Errorf("parse prometheus metrics: %w", fmt.Errorf("unexpected token")))}),
			want: "failed to scrape any Intel XPUMD exporter pod (check XPUMD pod status and network connectivity)",
		},
		{
			name: "intel list failure wrapping list pods",
			err:  fmt.Errorf("listing Intel XPUMD pods failed: %w", fmt.Errorf("list pods: %w", fmt.Errorf("forbidden"))),
			want: "failed to list Intel XPUMD exporter pods (RBAC/cluster connectivity issue)",
		},
		{
			name: "intel scrape failure wrapping no xpu devices reports the root cause",
			err:  fmt.Errorf("failed to scrape any Intel XPUMD exporter pod: %v", []error{fmt.Errorf("pod xpumd (10.0.0.2): %w", fmt.Errorf("no XPU devices detected from XPUMD metrics"))}),
			want: "no XPU devices detected in XPUMD metrics",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, GetGPUDiscoveryFailureReason(tc.err))
		})
	}
}

// TestGetGPUDiscoveryFailureReason_DCGMUnaffectedByIntelReorder ensures the
// DCGM/generic reasons are unchanged after the Intel-specific cases were moved
// ahead of them.
func TestGetGPUDiscoveryFailureReason_DCGMUnaffectedByIntelReorder(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "dcgm http get",
			err:  fmt.Errorf("HTTP GET %s failed: %w", "http://10.0.0.3:9400/metrics", fmt.Errorf("connection refused")),
			want: "failed to reach DCGM metrics endpoint on pod (network/port issue)",
		},
		{
			name: "dcgm list pods",
			err:  fmt.Errorf("list pods: %w", fmt.Errorf("forbidden")),
			want: "failed to list DCGM exporter pods (RBAC/cluster connectivity issue)",
		},
		{
			name: "dcgm not enabled",
			err:  fmt.Errorf("DCGM is not enabled in the GPU Operator (check GPU Operator configuration and permissions)"),
			want: "DCGM is not enabled in the GPU Operator (check GPU Operator configuration and permissions)",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, GetGPUDiscoveryFailureReason(tc.err))
		})
	}
}

func TestCreateProfilingJobWithManualHardwareDoesNotRequireAPIReader(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, batchv1.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "manual-hardware",
			Namespace: "default",
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "test-model",
			Backend: nvidiacomv1beta1.BackendTypeVllm,
			Image:   "test-profiler:latest",
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU:         nvidiacomv1beta1.GPUSKUTypeH100SXM,
				VRAMMB:         ptr.To(81920.0),
				NumGPUsPerNode: ptr.To(int32(8)),
				TotalGPUs:      ptr.To(int32(16)),
			},
		},
	}
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgdr).Build()
	r := &DynamoGraphDeploymentRequestReconciler{
		Client:      fakeClient,
		Recorder:    &record.FakeRecorder{},
		Config:      &configv1alpha1.OperatorConfiguration{},
		RBACManager: &MockRBACManager{},
	}

	var fetched nvidiacomv1beta1.DynamoGraphDeploymentRequest
	require.NoError(t, fakeClient.Get(ctx, types.NamespacedName{Name: dgdr.Name, Namespace: dgdr.Namespace}, &fetched))

	requeue, err := r.createProfilingJob(ctx, &fetched)
	require.NoError(t, err)
	require.True(t, requeue)

	requeue, err = r.createProfilingJob(ctx, &fetched)
	require.NoError(t, err)
	require.False(t, requeue)

	job := &batchv1.Job{}
	require.NoError(t, fakeClient.Get(ctx, types.NamespacedName{
		Name:      getProfilingJobName(&fetched),
		Namespace: fetched.Namespace,
	}, job))
}

// TestEnrichHardwareFromDiscovery_NormalizesBareModelFromDCGM is the regression test for
// the bug where DCGM reports "NVIDIA H200" (no SXM suffix, system="") and the controller
// serialized the raw string into the profiling job config instead of normalizing it to
// "h200_sxm", causing the Python profiler's Pydantic enum validation to fail.
func TestEnrichHardwareFromDiscovery_NormalizesBareModelFromDCGM(t *testing.T) {
	tests := []struct {
		name           string
		dcgmModel      string
		expectedGPUSKU string
	}{
		{
			name:           "NVIDIA H200 from DCGM normalizes to h200_sxm",
			dcgmModel:      "NVIDIA H200",
			expectedGPUSKU: "h200_sxm",
		},
		{
			name:           "NVIDIA B200 from DCGM normalizes to b200_sxm",
			dcgmModel:      "NVIDIA B200",
			expectedGPUSKU: "b200_sxm",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			_ = corev1.AddToScheme(scheme)

			dcgmPod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "dcgm-exporter",
					Namespace: "default",
					Labels: map[string]string{
						gpupkg.LabelApp: gpupkg.LabelValueNvidiaDCGMExporter,
					},
				},
				Status: corev1.PodStatus{
					Phase: corev1.PodRunning,
					PodIP: "10.0.0.1",
				},
			}
			fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dcgmPod).Build()

			// Mock scraper returns System="" to simulate the scenario where
			// DCGM metrics lack a form factor suffix (e.g. "NVIDIA H200").
			mockScraper := func(_ context.Context, _ string) (*gpupkg.GPUInfo, error) {
				return &gpupkg.GPUInfo{
					NodeName:    "gpu-node",
					GPUsPerNode: 8,
					Model:       tt.dcgmModel,
					VRAMPerGPU:  143770,
					System:      "",
				}, nil
			}

			r := &DynamoGraphDeploymentRequestReconciler{
				Client:            fakeClient,
				APIReader:         fakeClient,
				Recorder:          &record.FakeRecorder{},
				GPUDiscovery:      gpupkg.NewGPUDiscovery(mockScraper),
				GPUDiscoveryCache: gpupkg.NewGPUDiscoveryCache(),
			}

			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{},
			}

			_, err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
			require.NoError(t, err)
			require.NotNil(t, dgdr.Spec.Hardware)
			assert.Equal(t, tt.expectedGPUSKU, string(dgdr.Spec.Hardware.GPUSKU),
				"gpuSku must be a valid profiler enum, not the raw DCGM model string %q", tt.dcgmModel)
		})
	}
}

// TestEnrichHardwareFromDiscovery_FallsBackToModelForUnknownGPU verifies that for GPUs
// not in the AIC support matrix, the raw GFD product name is used as a fallback.
func TestEnrichHardwareFromDiscovery_FallsBackToModelForUnknownGPU(t *testing.T) {
	r := newFakeReconciler(gpuNode("gpu-node-1", "Tesla-V100-SXM2-16GB", 8, 16384))

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
			Hardware: &nvidiacomv1beta1.HardwareSpec{
				GPUSKU:         "Tesla-V100-SXM2-16GB",
				VRAMMB:         ptr.To(float64(16384)),
				NumGPUsPerNode: ptr.To(int32(8)),
				TotalGPUs:      ptr.To(int32(8)),
			},
		},
	}

	_, err := r.enrichHardwareFromDiscovery(context.Background(), dgdr)
	require.NoError(t, err)
	require.NotNil(t, dgdr.Spec.Hardware)
	assert.Equal(t, "Tesla-V100-SXM2-16GB", string(dgdr.Spec.Hardware.GPUSKU),
		"Unknown GPU should fall back to raw model name")
}
