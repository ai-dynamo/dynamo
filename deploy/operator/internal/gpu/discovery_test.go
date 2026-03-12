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
	"testing"

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
	assert.Equal(t, "h100_sxm", string(gpuInfo.System))
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
		// GFD product names as seen in real cluster labels (regression for GPUSKU bug)
		{"NVIDIA-B200", "b200_sxm", "B200 with NVIDIA prefix (GFD label format)"},
		{"NVIDIA-H200-SXM5-141GB", "h200_sxm", "H200 with NVIDIA prefix (GFD label format)"},
	}

	for _, tt := range tests {
		t.Run(tt.description, func(t *testing.T) {
			result := InferHardwareSystem(tt.gpuProduct)
			assert.Equal(t, tt.expectedSystem, string(result), "Failed for GPU: %s", tt.gpuProduct)
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
		assert.Equal(t, "h100_sxm", string(result), "Should handle case variations: %s", variant)
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
		assert.Equal(t, "h100_sxm", string(result), "Should normalize spaces/dashes: %s", variant)
	}
}

func TestDiscoverGPUs_CollectsNodeTaints(t *testing.T) {
	ctx := context.Background()

	gpuNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-node-1",
			Labels: map[string]string{
				LabelGPUCount:   "8",
				LabelGPUProduct: "H100-SXM5-80GB",
				LabelGPUMemory:  "81920",
			},
		},
		Spec: corev1.NodeSpec{
			Taints: []corev1.Taint{
				{Key: "nvidia.com/gpu", Effect: corev1.TaintEffectNoSchedule},
				{Key: "dedicated", Value: "user-workload", Effect: corev1.TaintEffectNoExecute},
			},
		},
	}
	cpuNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "cpu-node-1",
			Labels: map[string]string{},
		},
		Spec: corev1.NodeSpec{
			Taints: []corev1.Taint{
				{Key: "team", Effect: corev1.TaintEffectNoSchedule},
			},
		},
	}

	k8sClient := newFakeClient(gpuNode, cpuNode)
	gpuInfo, err := DiscoverGPUs(ctx, k8sClient)
	require.NoError(t, err)
	require.NotNil(t, gpuInfo)

	// Should collect taints from both GPU and CPU nodes
	assert.Len(t, gpuInfo.NodeTaints, 3)
}

func TestDiscoverNodeTaints(t *testing.T) {
	ctx := context.Background()

	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
		Spec: corev1.NodeSpec{
			Taints: []corev1.Taint{
				{Key: "nvidia.com/gpu", Effect: corev1.TaintEffectNoSchedule},
				{Key: "dedicated", Value: "user-workload", Effect: corev1.TaintEffectNoExecute},
			},
		},
	}
	node2 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-2"},
		Spec: corev1.NodeSpec{
			Taints: []corev1.Taint{
				// Duplicate of node1's first taint
				{Key: "nvidia.com/gpu", Effect: corev1.TaintEffectNoSchedule},
				{Key: "kubernetes.io/arch", Effect: corev1.TaintEffectNoSchedule},
			},
		},
	}

	k8sClient := newFakeClient(node1, node2)
	taints, err := DiscoverNodeTaints(ctx, k8sClient)
	require.NoError(t, err)

	// Should deduplicate: nvidia.com/gpu appears once, not twice
	assert.Len(t, taints, 3)
}

func TestTaintsToTolerations(t *testing.T) {
	taints := []corev1.Taint{
		{Key: "nvidia.com/gpu", Effect: corev1.TaintEffectNoSchedule},
		{Key: "dedicated", Value: "user-workload", Effect: corev1.TaintEffectNoExecute},
	}

	tolerations := TaintsToTolerations(taints)
	require.Len(t, tolerations, 2)

	// Empty-value taint → Exists operator
	assert.Equal(t, "nvidia.com/gpu", tolerations[0].Key)
	assert.Equal(t, corev1.TolerationOpExists, tolerations[0].Operator)
	assert.Equal(t, corev1.TaintEffectNoSchedule, tolerations[0].Effect)
	assert.Empty(t, tolerations[0].Value)

	// Valued taint → Equal operator
	assert.Equal(t, "dedicated", tolerations[1].Key)
	assert.Equal(t, corev1.TolerationOpEqual, tolerations[1].Operator)
	assert.Equal(t, "user-workload", tolerations[1].Value)
	assert.Equal(t, corev1.TaintEffectNoExecute, tolerations[1].Effect)
}

func TestTaintsToTolerations_Empty(t *testing.T) {
	tolerations := TaintsToTolerations(nil)
	assert.Empty(t, tolerations)
}
