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

package gpu

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// newFakeClient creates a fake Kubernetes client with the given objects
func newFakeClient(objs ...client.Object) client.Client {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	return fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		Build()
}

func TestDiscoverGPUs_SingleNode(t *testing.T) {
	ctx := context.Background()

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-node-1",
			Labels: map[string]string{
				LabelGPUCount:   "8",
				LabelGPUProduct: "H100-SXM5-80GB",
				LabelGPUMemory:  "81920",
			},
		},
	}

	k8sClient := newFakeClient(node)

	gpuInfo, err := DiscoverGPUs(ctx, k8sClient)
	require.NoError(t, err)
	require.NotNil(t, gpuInfo)

	assert.Equal(t, 8, gpuInfo.GPUsPerNode)
	assert.Equal(t, "H100-SXM5-80GB", gpuInfo.Model)
	assert.Equal(t, 81920, gpuInfo.VRAMPerGPU)
	assert.Equal(t, "h100_sxm", gpuInfo.System)
}

func TestDiscoverGPUs_MultipleNodesHomogeneous(t *testing.T) {
	ctx := context.Background()

	// Multiple nodes with same GPU configuration
	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-node-1",
			Labels: map[string]string{
				LabelGPUCount:   "8",
				LabelGPUProduct: "H100-SXM5-80GB",
				LabelGPUMemory:  "81920",
			},
		},
	}
	node2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-node-2",
			Labels: map[string]string{
				LabelGPUCount:   "8",
				LabelGPUProduct: "H100-SXM5-80GB",
				LabelGPUMemory:  "81920",
			},
		},
	}

	k8sClient := newFakeClient(node1, node2)

	gpuInfo, err := DiscoverGPUs(ctx, k8sClient)
	require.NoError(t, err)
	require.NotNil(t, gpuInfo)

	assert.Equal(t, 8, gpuInfo.GPUsPerNode)
	assert.Equal(t, "H100-SXM5-80GB", gpuInfo.Model)
	assert.Equal(t, 81920, gpuInfo.VRAMPerGPU)
}

func TestDiscoverGPUs_MultipleNodesHeterogeneous_HigherGPUCountWins(t *testing.T) {
	ctx := context.Background()

	// Node with fewer GPUs
	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-node-1",
			Labels: map[string]string{
				LabelGPUCount:   "4",
				LabelGPUProduct: "A100-SXM4-40GB",
				LabelGPUMemory:  "40960",
			},
		},
	}

	// Node with more GPUs (should win)
	node2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-node-2",
			Labels: map[string]string{
				LabelGPUCount:   "8",
				LabelGPUProduct: "H100-SXM5-80GB",
				LabelGPUMemory:  "81920",
			},
		},
	}

	k8sClient := newFakeClient(node1, node2)

	gpuInfo, err := DiscoverGPUs(ctx, k8sClient)
	require.NoError(t, err)
	require.NotNil(t, gpuInfo)

	// Should prefer node with 8 GPUs over node with 4 GPUs
	assert.Equal(t, 8, gpuInfo.GPUsPerNode)
	assert.Equal(t, "H100-SXM5-80GB", gpuInfo.Model)
	assert.Equal(t, 81920, gpuInfo.VRAMPerGPU)
}

func TestDiscoverGPUs_MultipleNodesHeterogeneous_HigherVRAMWins(t *testing.T) {
	ctx := context.Background()

	// Node with same GPU count but less VRAM
	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-node-1",
			Labels: map[string]string{
				LabelGPUCount:   "8",
				LabelGPUProduct: "A100-SXM4-40GB",
				LabelGPUMemory:  "40960",
			},
		},
	}

	// Node with same GPU count but more VRAM (should win)
	node2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-node-2",
			Labels: map[string]string{
				LabelGPUCount:   "8",
				LabelGPUProduct: "H100-SXM5-80GB",
				LabelGPUMemory:  "81920",
			},
		},
	}

	k8sClient := newFakeClient(node1, node2)

	gpuInfo, err := DiscoverGPUs(ctx, k8sClient)
	require.NoError(t, err)
	require.NotNil(t, gpuInfo)

	// Should prefer node with higher VRAM when GPU count is equal
	assert.Equal(t, 8, gpuInfo.GPUsPerNode)
	assert.Equal(t, "H100-SXM5-80GB", gpuInfo.Model)
	assert.Equal(t, 81920, gpuInfo.VRAMPerGPU)
}

func TestDiscoverGPUs_MixedNodesWithAndWithoutGPUs(t *testing.T) {
	ctx := context.Background()

	// CPU-only node (no GPU labels)
	cpuNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "cpu-node-1",
			Labels: map[string]string{},
		},
	}

	// GPU node
	gpuNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-node-1",
			Labels: map[string]string{
				LabelGPUCount:   "8",
				LabelGPUProduct: "H100-SXM5-80GB",
				LabelGPUMemory:  "81920",
			},
		},
	}

	k8sClient := newFakeClient(cpuNode, gpuNode)

	gpuInfo, err := DiscoverGPUs(ctx, k8sClient)
	require.NoError(t, err)
	require.NotNil(t, gpuInfo)

	// Should find the GPU node and ignore CPU-only node
	assert.Equal(t, 8, gpuInfo.GPUsPerNode)
	assert.Equal(t, "H100-SXM5-80GB", gpuInfo.Model)
}

func TestDiscoverGPUs_NoNodes(t *testing.T) {
	ctx := context.Background()
	k8sClient := newFakeClient() // Empty cluster

	gpuInfo, err := DiscoverGPUs(ctx, k8sClient)
	assert.Error(t, err)
	assert.Nil(t, gpuInfo)
	assert.Contains(t, err.Error(), "no nodes found")
}

func TestDiscoverGPUs_NoGPUNodes(t *testing.T) {
	ctx := context.Background()

	// Only CPU nodes
	cpuNode1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "cpu-node-1",
			Labels: map[string]string{},
		},
	}
	cpuNode2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "cpu-node-2",
			Labels: map[string]string{
				"node-type": "cpu-only",
			},
		},
	}

	k8sClient := newFakeClient(cpuNode1, cpuNode2)

	gpuInfo, err := DiscoverGPUs(ctx, k8sClient)
	assert.Error(t, err)
	assert.Nil(t, gpuInfo)
	assert.Contains(t, err.Error(), "no nodes with NVIDIA GPU Feature Discovery labels found")
}

func TestExtractGPUInfoFromNode_MissingLabels(t *testing.T) {
	tests := []struct {
		name        string
		labels      map[string]string
		expectError bool
		errorMsg    string
	}{
		{
			name:        "missing GPU count",
			labels:      map[string]string{LabelGPUProduct: "H100", LabelGPUMemory: "80000"},
			expectError: true,
			errorMsg:    LabelGPUCount,
		},
		{
			name:        "missing GPU product",
			labels:      map[string]string{LabelGPUCount: "8", LabelGPUMemory: "80000"},
			expectError: true,
			errorMsg:    LabelGPUProduct,
		},
		{
			name:        "missing GPU memory",
			labels:      map[string]string{LabelGPUCount: "8", LabelGPUProduct: "H100"},
			expectError: true,
			errorMsg:    LabelGPUMemory,
		},
		{
			name:        "invalid GPU count",
			labels:      map[string]string{LabelGPUCount: "invalid", LabelGPUProduct: "H100", LabelGPUMemory: "80000"},
			expectError: true,
			errorMsg:    "invalid GPU count",
		},
		{
			name:        "invalid GPU memory",
			labels:      map[string]string{LabelGPUCount: "8", LabelGPUProduct: "H100", LabelGPUMemory: "invalid"},
			expectError: true,
			errorMsg:    "invalid GPU memory",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test-node",
					Labels: tt.labels,
				},
			}

			gpuInfo, err := extractGPUInfoFromNode(node)
			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, gpuInfo)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, gpuInfo)
			}
		})
	}
}

func TestInferHardwareSystem(t *testing.T) {
	tests := []struct {
		gpuProduct     string
		expectedSystem string
		description    string
	}{
		{"H100-SXM5-80GB", "h100_sxm", "H100 SXM variant"},
		{"H100-PCIE-80GB", "h100_sxm", "H100 PCIe variant (mapped to SXM)"},
		{"H200-SXM5-141GB", "h200_sxm", "H200 SXM variant"},
		{"A100-SXM4-40GB", "a100_sxm", "A100 SXM variant"},
		{"A100-PCIE-80GB", "a100_sxm", "A100 PCIe variant (mapped to SXM)"},
		{"L40S", "l40s", "L40S"},
		{"NVIDIA L40S", "l40s", "L40S with prefix"},
		{"B200-SXM", "b200_sxm", "B200 SXM"},
		{"GB200", "gb200_sxm", "GB200"},
		{"Tesla V100-SXM2-16GB", "", "V100 (not in mapping)"},
		{"RTX 4090", "", "Consumer GPU (not in mapping)"},
		{"Unknown-GPU", "", "Unknown GPU"},
		{"", "", "Empty string"},
	}

	for _, tt := range tests {
		t.Run(tt.description, func(t *testing.T) {
			result := InferHardwareSystem(tt.gpuProduct)
			assert.Equal(t, tt.expectedSystem, result, "Failed for GPU: %s", tt.gpuProduct)
		})
	}
}

func TestInferHardwareSystem_CaseInsensitive(t *testing.T) {
	// Test that inference is case-insensitive
	variants := []string{
		"h100-sxm5-80gb",
		"H100-SXM5-80GB",
		"H100-sxm5-80GB",
		"h100-SXM5-80gb",
	}

	for _, variant := range variants {
		result := InferHardwareSystem(variant)
		assert.Equal(t, "h100_sxm", result, "Should handle case variations: %s", variant)
	}
}

func TestInferHardwareSystem_SpacesAndDashes(t *testing.T) {
	// Test that spaces and dashes are normalized
	variants := []string{
		"H100-SXM5-80GB",
		"H100 SXM5 80GB",
		"H100SXM580GB",
		"H100-SXM5 80GB",
	}

	for _, variant := range variants {
		result := InferHardwareSystem(variant)
		assert.Equal(t, "h100_sxm", result, "Should normalize spaces/dashes: %s", variant)
	}
}

func TestParseMetrics(t *testing.T) {
	ctx := context.Background()

	// Fake DCGM metrics for a node with 2 GPUs
	metricFamilies := map[string]*dto.MetricFamily{
		"DCGM_FI_DEV_GPU_TEMP": {
			Metric: []*dto.Metric{
				{
					Label: []*dto.LabelPair{
						{Name: strPtr("gpu"), Value: strPtr("0")},
						{Name: strPtr("modelName"), Value: strPtr("H100-SXM5-80GB")},
						{Name: strPtr("Hostname"), Value: strPtr("node1")},
					},
				},
				{
					Label: []*dto.LabelPair{
						{Name: strPtr("gpu"), Value: strPtr("1")},
						{Name: strPtr("modelName"), Value: strPtr("H100-SXM5-80GB")},
						{Name: strPtr("Hostname"), Value: strPtr("node1")},
					},
				},
			},
		},
		"DCGM_FI_DEV_FB_FREE": {
			Metric: []*dto.Metric{
				{Label: []*dto.LabelPair{{Name: strPtr("gpu"), Value: strPtr("0")}}, Gauge: &dto.Gauge{Value: float64Ptr(10000)}},
				{Label: []*dto.LabelPair{{Name: strPtr("gpu"), Value: strPtr("1")}}, Gauge: &dto.Gauge{Value: float64Ptr(12000)}},
			},
		},
		"DCGM_FI_DEV_FB_USED": {
			Metric: []*dto.Metric{
				{Label: []*dto.LabelPair{{Name: strPtr("gpu"), Value: strPtr("0")}}, Gauge: &dto.Gauge{Value: float64Ptr(5000)}},
				{Label: []*dto.LabelPair{{Name: strPtr("gpu"), Value: strPtr("1")}}, Gauge: &dto.Gauge{Value: float64Ptr(6000)}},
			},
		},
		"DCGM_FI_DEV_FB_RESERVED": {
			Metric: []*dto.Metric{
				{Label: []*dto.LabelPair{{Name: strPtr("gpu"), Value: strPtr("0")}}, Gauge: &dto.Gauge{Value: float64Ptr(0)}},
				{Label: []*dto.LabelPair{{Name: strPtr("gpu"), Value: strPtr("1")}}, Gauge: &dto.Gauge{Value: float64Ptr(0)}},
			},
		},
	}

	info, err := parseMetrics(ctx, metricFamilies)
	require.NoError(t, err)

	assert.Equal(t, "node1", info.NodeName)
	assert.Equal(t, 2, info.GPUsPerNode)
	assert.Equal(t, "H100-SXM5-80GB", info.Model)
	// maxVRAM: 12000 + 6000 + 0 = 18000
	assert.Equal(t, 18000, info.VRAMPerGPU)
	assert.False(t, info.MIGEnabled)
	assert.Empty(t, info.MIGProfiles)
}

func TestScrapeMetricsEndpoint(t *testing.T) {
	// Create a test HTTP server to simulate DCGM exporter
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `
# HELP DCGM_FI_DEV_GPU_TEMP Dummy temperature metric
# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{gpu="0",modelName="H100-SXM5-80GB",Hostname="node1"} 50
`)
	}))
	defer ts.Close()

	ctx := context.Background()
	info, err := scrapeMetricsEndpoint(ctx, ts.URL)
	require.NoError(t, err)

	assert.Equal(t, "node1", info.NodeName)
	assert.Equal(t, 1, info.GPUsPerNode)
	assert.Equal(t, "H100-SXM5-80GB", info.Model)
}

func TestDiscoverGPUsFromDCGM_CacheHit(t *testing.T) {
	ctx := context.Background()

	// Fake pod representing DCGM exporter
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcgm-pod",
			Namespace: "default",
			Labels: map[string]string{
				LabelApp: LabelValueNvidiaDCGMExporter,
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			PodIP: "10.0.0.1",
		},
	}

	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)

	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(pod).
		Build()

	cache := &GPUDiscoveryCache{}

	// Track number of times scrape is called
	callCount := 0

	// Override scrape function
	originalScrape := scrapeMetricsFunc
	defer func() { scrapeMetricsFunc = originalScrape }()

	scrapeMetricsFunc = func(ctx context.Context, endpoint string) (*GPUInfo, error) {
		callCount++
		return &GPUInfo{
			NodeName:    "node-a",
			GPUsPerNode: 4,
			Model:       "A100",
			VRAMPerGPU:  40960,
			MIGEnabled:  false,
			MIGProfiles: map[string]int{},
			System:      "DGX",
		}, nil
	}

	// First call → should scrape
	info1, err := DiscoverGPUsFromDCGM(ctx, k8sClient, cache)
	require.NoError(t, err)
	require.NotNil(t, info1)
	require.Equal(t, 1, callCount)

	// Second call → should hit cache
	info2, err := DiscoverGPUsFromDCGM(ctx, k8sClient, cache)
	require.NoError(t, err)
	require.NotNil(t, info2)

	// Scrape should NOT be called again
	require.Equal(t, 1, callCount)

	// Results should be identical
	require.Equal(t, info1, info2)
}

func TestDiscoverGPUsFromDCGM_GPUOperatorInstalled_DCgmNotEnabled(t *testing.T) {
	ctx := context.Background()

	// Fake running GPU Operator pod
	gpuOperatorPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "gpu-operator-abc",
			Namespace: "gpu-operator",
			Labels: map[string]string{
				LabelApp: LabelValueGPUOperator,
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
		},
	}

	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))

	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(gpuOperatorPod).
		Build()

	cache := NewGPUDiscoveryCache()

	info, err := DiscoverGPUsFromDCGM(ctx, k8sClient, cache)

	require.Nil(t, info)
	require.Error(t, err)
	require.Contains(t, err.Error(), "DCGM is not enabled in the GPU Operator")
}

func TestDiscoverGPUsFromDCGM_NoGPUOperator_NoDCGM(t *testing.T) {
	ctx := context.Background()

	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))

	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()

	cache := NewGPUDiscoveryCache()

	info, err := DiscoverGPUsFromDCGM(ctx, k8sClient, cache)

	require.Nil(t, info)
	require.Error(t, err)

	require.True(
		t,
		strings.Contains(err.Error(), "no DCGM exporter pods found"),
		"expected no DCGM exporter pods error",
	)
}

func TestListDCGMExporterPods(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)

	ctx := context.Background()

	tests := []struct {
		name        string
		objects     []client.Object
		expectCount int
		expectErr   bool
		errorClient bool
	}{
		{
			name: "pods found via different selectors",
			objects: []client.Object{
				&corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod1",
						Namespace: "ns1",
						Labels: map[string]string{
							LabelApp: LabelValueNvidiaDCGMExporter,
						},
					},
				},
				&corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod2",
						Namespace: "ns1",
						Labels: map[string]string{
							LabelAppKubernetesName: LabelValueDCGMExporter,
						},
					},
				},
			},
			expectCount: 2,
			expectErr:   false,
		},
		{
			name: "duplicate pods across selectors should dedupe",
			objects: []client.Object{
				&corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pod1",
						Namespace: "ns1",
						Labels: map[string]string{
							LabelApp:               LabelValueDCGMExporter,
							LabelAppKubernetesName: LabelValueDCGMExporter,
						},
					},
				},
			},
			expectCount: 1,
			expectErr:   false,
		},
		{
			name:        "no pods found",
			objects:     []client.Object{},
			expectCount: 0,
			expectErr:   true,
		},
		{
			name:        "client list error",
			objects:     []client.Object{},
			expectCount: 0,
			expectErr:   true,
			errorClient: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			var k8sClient client.Client

			if tt.errorClient {
				k8sClient = &errorListClient{}
			} else {
				k8sClient = fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(tt.objects...).
					Build()
			}

			pods, err := listDCGMExporterPods(ctx, k8sClient)

			if tt.expectErr && err == nil {
				t.Fatalf("expected error but got nil")
			}
			if !tt.expectErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(pods) != tt.expectCount {
				t.Fatalf("expected %d pods, got %d", tt.expectCount, len(pods))
			}
		})
	}
}

package cloud_test

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	. "your/module/cloud" // replace with the package where GetCloudProviderInfo is
)

func TestGetCloudProviderInfo(t *testing.T) {
	scheme := runtime.NewScheme()
	corev1.AddToScheme(scheme)

	tests := []struct {
		name       string
		nodes      []corev1.Node
		want       string
		expectErr  bool
	}{
		{
			name:      "no nodes",
			nodes:     []corev1.Node{},
			want:      "unknown",
			expectErr: true,
		},
		{
			name: "AWS providerID detection",
			nodes: []corev1.Node{
				{
					Spec: corev1.NodeSpec{
						ProviderID: "aws:///us-east-1/i-1234567890abcdef0",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name: "node1",
					},
				},
			},
			want:      "aws",
			expectErr: false,
		},
		{
			name: "AKS label detection",
			nodes: []corev1.Node{
				{
					Spec: corev1.NodeSpec{
						ProviderID: "unknown",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name: "node1",
						Labels: map[string]string{
							"kubernetes.azure.com/cluster": "my-aks-cluster",
						},
					},
				},
			},
			want:      "aks",
			expectErr: false,
		},
		{
			name: "GCP instance type detection",
			nodes: []corev1.Node{
				{
					Spec: corev1.NodeSpec{
						ProviderID: "unknown",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name: "node1",
						Labels: map[string]string{
							"node.kubernetes.io/instance-type": "a2-standard-4",
						},
					},
				},
			},
			want:      "gcp",
			expectErr: false,
		},
		{
			name: "unknown provider",
			nodes: []corev1.Node{
				{
					Spec: corev1.NodeSpec{
						ProviderID: "my-custom-cloud",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:   "node1",
						Labels: map[string]string{},
					},
				},
			},
			want:      "other",
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clientBuilder := fake.NewClientBuilder().WithScheme(scheme).WithObjects()
			for _, node := range tt.nodes {
				clientBuilder.WithObjects(&node)
			}
			k8sClient := clientBuilder.Build()

			got, err := GetCloudProviderInfo(context.Background(), k8sClient)
			if (err != nil) != tt.expectErr {
				t.Errorf("unexpected error status: got %v, want error: %v", err, tt.expectErr)
			}
			if !strings.EqualFold(got, tt.want) {
				t.Errorf("unexpected provider: got %s, want %s", got, tt.want)
			}
		})
	}
}
//
// ---- Fake client that forces List error ----
//

type errorListClient struct {
	client.Client
}

func (e *errorListClient) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	return errors.New("forced list error")
}

// --- Helper functions ---
func strPtr(s string) *string       { return &s }
func float64Ptr(f float64) *float64 { return &f }
