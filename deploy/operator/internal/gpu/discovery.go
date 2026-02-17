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
	"fmt"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// NVIDIA GPU Feature Discovery (GFD) label keys
	LabelGPUCount   = "nvidia.com/gpu.count"
	LabelGPUProduct = "nvidia.com/gpu.product"
	LabelGPUMemory  = "nvidia.com/gpu.memory"
)

// GPUInfo contains discovered GPU configuration from cluster nodes
type GPUInfo struct {
	GPUsPerNode int    // Maximum GPUs per node found in the cluster
	Model       string // GPU product name (e.g., "H100-SXM5-80GB")
	VRAMPerGPU  int    // VRAM in MiB per GPU
	System      string // AIC hardware system identifier (e.g., "h100_sxm", "h200_sxm"), empty if unknown
}

// DiscoverGPUs queries Kubernetes nodes to determine GPU configuration.
// It extracts GPU information from NVIDIA GPU Feature Discovery (GFD) labels
// and returns aggregated GPU info, preferring nodes with higher GPU count,
// then higher VRAM if counts are equal.
//
// This function requires cluster-wide node read permissions and expects nodes
// to have GFD labels. If no nodes with GPU labels are found, it returns an error.
func DiscoverGPUs(ctx context.Context, k8sClient client.Client) (*GPUInfo, error) {
	logger := log.FromContext(ctx)
	logger.Info("Starting GPU discovery from cluster nodes")

	// List all nodes in the cluster
	nodeList := &corev1.NodeList{}
	if err := k8sClient.List(ctx, nodeList); err != nil {
		return nil, fmt.Errorf("failed to list cluster nodes: %w", err)
	}

	if len(nodeList.Items) == 0 {
		return nil, fmt.Errorf("no nodes found in cluster")
	}

	logger.Info("Found cluster nodes", "count", len(nodeList.Items))

	// Track the best GPU configuration found
	var bestGPUInfo *GPUInfo
	nodesWithGPUs := 0

	for i := range nodeList.Items {
		node := &nodeList.Items[i]
		gpuInfo, err := extractGPUInfoFromNode(node)
		if err != nil {
			// Node doesn't have GPU labels or has invalid labels, skip it
			logger.V(1).Info("Skipping node without valid GPU info",
				"node", node.Name,
				"reason", err.Error())
			continue
		}

		nodesWithGPUs++
		logger.Info("Found GPU node",
			"node", node.Name,
			"gpus", gpuInfo.GPUsPerNode,
			"model", gpuInfo.Model,
			"vram", gpuInfo.VRAMPerGPU)

		// Select best configuration: prefer higher GPU count, then higher VRAM
		if bestGPUInfo == nil ||
			gpuInfo.GPUsPerNode > bestGPUInfo.GPUsPerNode ||
			(gpuInfo.GPUsPerNode == bestGPUInfo.GPUsPerNode && gpuInfo.VRAMPerGPU > bestGPUInfo.VRAMPerGPU) {
			bestGPUInfo = gpuInfo
		}
	}

	if bestGPUInfo == nil {
		return nil, fmt.Errorf("no nodes with NVIDIA GPU Feature Discovery labels found (checked %d nodes). "+
			"Ensure GPU nodes have labels: %s, %s, %s",
			len(nodeList.Items), LabelGPUCount, LabelGPUProduct, LabelGPUMemory)
	}

	// Infer hardware system from GPU model
	bestGPUInfo.System = InferHardwareSystem(bestGPUInfo.Model)

	logger.Info("GPU discovery completed",
		"gpusPerNode", bestGPUInfo.GPUsPerNode,
		"model", bestGPUInfo.Model,
		"vram", bestGPUInfo.VRAMPerGPU,
		"system", bestGPUInfo.System,
		"nodesWithGPUs", nodesWithGPUs)

	return bestGPUInfo, nil
}

// extractGPUInfoFromNode extracts GPU information from a single node's labels.
// Returns error if required labels are missing or invalid.
func extractGPUInfoFromNode(node *corev1.Node) (*GPUInfo, error) {
	labels := node.Labels
	if labels == nil {
		return nil, fmt.Errorf("node has no labels")
	}

	gpuCountStr, ok := labels[LabelGPUCount]
	if !ok {
		return nil, fmt.Errorf("missing label %s", LabelGPUCount)
	}
	gpuCount, err := strconv.Atoi(gpuCountStr)
	if err != nil || gpuCount <= 0 {
		return nil, fmt.Errorf("invalid GPU count: %s", gpuCountStr)
	}

	gpuModel, ok := labels[LabelGPUProduct]
	if !ok || gpuModel == "" {
		return nil, fmt.Errorf("missing or empty label %s", LabelGPUProduct)
	}

	// Extract VRAM (memory in MiB)
	gpuMemoryStr, ok := labels[LabelGPUMemory]
	if !ok {
		return nil, fmt.Errorf("missing label %s", LabelGPUMemory)
	}
	gpuMemory, err := strconv.Atoi(gpuMemoryStr)
	if err != nil || gpuMemory <= 0 {
		return nil, fmt.Errorf("invalid GPU memory: %s", gpuMemoryStr)
	}

	return &GPUInfo{
		GPUsPerNode: gpuCount,
		Model:       gpuModel,
		VRAMPerGPU:  gpuMemory,
	}, nil
}

// InferHardwareSystem maps GPU product name to hardware system identifier.
// Returns empty string if the GPU model cannot be confidently mapped.
//
// This is a best-effort mapping based on common NVIDIA datacenter GPU naming patterns.
// The system identifier is used by the profiler for performance estimation and configuration.
//
// Limitations:
//   - Cannot distinguish SXM vs. PCIe variants from labels alone (assumes SXM for datacenter GPUs)
//   - New GPU models require code updates (gracefully returns empty string)
//   - Non-standard SKU names may not match
//
// Users can manually override the system in their profiling config (hardware.system)
// if auto-detection is incorrect or unavailable.
func InferHardwareSystem(gpuProduct string) string {
	if gpuProduct == "" {
		return ""
	}

	// Normalize: uppercase, remove spaces/dashes for pattern matching
	normalized := strings.ToUpper(strings.ReplaceAll(gpuProduct, "-", ""))
	normalized = strings.ReplaceAll(normalized, " ", "")

	// Map common NVIDIA datacenter GPU products to hardware system identifiers
	patterns := []struct {
		pattern string
		system  string
	}{
		{"GB200", "gb200_sxm"},
		{"H200", "h200_sxm"},
		{"H100", "h100_sxm"},
		{"B200", "b200_sxm"},
		{"A100", "a100_sxm"},
		{"L40S", "l40s"},
	}

	for _, p := range patterns {
		if strings.Contains(normalized, p.pattern) {
			return p.system
		}
	}

	// Unknown GPU type, return empty string
	// User must specify system manually in profiling config (hardware.system)
	return ""
}
