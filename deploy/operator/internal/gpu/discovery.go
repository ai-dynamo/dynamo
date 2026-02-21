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
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"

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
	NodeName    string         // Name of the node with this GPU configuration
	GPUsPerNode int            // Maximum GPUs per node found in the cluster
	Model       string         // GPU product name (e.g., "H100-SXM5-80GB")
	VRAMPerGPU  int            // VRAM in MiB per GPU
	System      string         // AIC hardware system identifier (e.g., "h100_sxm", "h200_sxm"), empty if unknown
	MIGEnabled  bool           // True if MIG is enabled (inferred from model or additional labels, not implemented in this version)
	MIGProfiles map[string]int // Optional: map of MIG profile name to count (requires additional label parsing, not implemented in this version)
}

type GPUDiscoveryCache struct {
	mu        sync.RWMutex
	value     *GPUInfo
	expiresAt time.Time
}

var scrapePodFunc = scrapePod

// NewGPUDiscoveryCache creates a new GPUDiscoveryCache instance.
//
// The cache stores a single discovered GPUInfo value with an expiration time.
// It is safe for concurrent use and is intended to reduce repeated DCGM
// scraping during reconciliation loops.
func NewGPUDiscoveryCache() *GPUDiscoveryCache {
	return &GPUDiscoveryCache{}
}

// Get returns the cached GPUInfo if it exists and has not expired.
//
// The boolean return value indicates whether a valid cached value was found.
// If the cache is empty or expired, it returns (nil, false).
//
// This method is safe for concurrent use.
func (c *GPUDiscoveryCache) Get() (*GPUInfo, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if time.Now().Before(c.expiresAt) && c.value != nil {
		return c.value, true
	}
	return nil, false
}

// Set stores the provided GPUInfo in the cache with the given TTL (time-to-live).
//
// The cached value will be considered valid until the TTL duration elapses.
// After expiration, Get will return (nil, false) until a new value is set.
//
// This method is safe for concurrent use.
func (c *GPUDiscoveryCache) Set(info *GPUInfo, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.value = info
	c.expiresAt = time.Now().Add(ttl)
}

// DiscoverGPUsFromDCGM discovers GPU configuration by scraping DCGM exporter pods.
//
// The function:
//   - First checks the provided cache for a valid, non-expired result.
//   - Lists pods labeled "app=dcgm-exporter".
//   - Installs DCGM via Helm if no exporter pods are found.
//   - Scrapes metrics from each exporter pod concurrently.
//   - Selects the "best" GPU configuration (highest GPU count, then highest VRAM).
//   - Stores the result in the cache for a short duration.
//
// This function does not require node read permissions, but it does require:
//   - Permission to list pods
//   - Network access to DCGM exporter metrics endpoints
//
// Returns an error if no valid GPU metrics can be obtained.
func DiscoverGPUsFromDCGM(ctx context.Context, k8sClient client.Client, cache *GPUDiscoveryCache) (*GPUInfo, error) {
	if cached, ok := cache.Get(); ok {
		return cached, nil
	}

	podList := &corev1.PodList{}
	if err := k8sClient.List(ctx, podList,
		client.MatchingLabels{"app": "dcgm-exporter"},
	); err != nil {
		return nil, fmt.Errorf("list dcgm exporter pods: %w", err)
	}

	if len(podList.Items) == 0 {
		if err := EnsureDCGMHelmRelease(ctx, "gpu-system"); err != nil {
			return nil, fmt.Errorf("install dcgm via helm: %w", err)
		}

		if err := WaitForDCGMPods(ctx, k8sClient, 3*time.Minute); err != nil {
			return nil, err
		}

		if err := k8sClient.List(ctx, podList,
			client.MatchingLabels{"app": "dcgm-exporter"},
		); err != nil {
			return nil, err
		}
	}

	var (
		best *GPUInfo
		mu   sync.Mutex
		wg   sync.WaitGroup
	)

	for i := range podList.Items {
		pod := podList.Items[i]
		if pod.Status.PodIP == "" {
			continue
		}

		wg.Add(1)
		go func(p corev1.Pod) {
			defer wg.Done()

			info, err := scrapePodFunc(ctx, p)
			if err != nil {
				return
			}

			mu.Lock()
			defer mu.Unlock()

			if best == nil ||
				info.GPUsPerNode > best.GPUsPerNode ||
				(info.GPUsPerNode == best.GPUsPerNode &&
					info.VRAMPerGPU > best.VRAMPerGPU) {
				best = info
			}
		}(pod)
	}

	wg.Wait()

	if best == nil {
		return nil, fmt.Errorf("no valid GPU info found")
	}

	cache.Set(best, 60*time.Second)
	return best, nil
}

// scrapePod retrieves and parses Prometheus metrics from a single DCGM exporter pod.
//
// It performs an HTTP GET request against the pod's metrics endpoint
// (http://<podIP>:9400/metrics), parses the Prometheus text format,
// and extracts GPU information using parseMetrics.
//
// Returns an error if the metrics endpoint is unreachable or parsing fails.
func scrapePod(ctx context.Context, pod corev1.Pod) (*GPUInfo, error) {
	url := fmt.Sprintf("http://%s:9400/metrics", pod.Status.PodIP)

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	parser := expfmt.TextParser{}
	metricFamilies, err := parser.TextToMetricFamilies(resp.Body)
	if err != nil {
		return nil, err
	}

	return parseMetrics(pod.Spec.NodeName, metricFamilies)
}

// parseMetrics extracts GPU information from parsed Prometheus metric families.
//
// It reads relevant DCGM metrics including:
//   - DCGM_FI_DEV_COUNT (GPU count)
//   - DCGM_FI_DEV_NAME (GPU model)
//   - DCGM_FI_DEV_FB_TOTAL (framebuffer memory)
//   - DCGM_FI_DEV_MIG_MODE (MIG enabled state)
//   - DCGM_FI_DEV_MIG_PROFILE (MIG profile distribution)
//
// The function returns a GPUInfo structure populated with the extracted data.
// If no GPUs are detected (count == 0), it returns an error.
//
// Assumptions and limitations:
//   - Uses the first metric sample for device-level metrics.
//   - Does not aggregate heterogeneous GPU configurations across nodes.
//   - Expects DCGM exporter to expose standard metric names.
func parseMetrics(node string, families map[string]*dto.MetricFamily) (*GPUInfo, error) {
	var (
		gpuCount    int
		model       string
		vram        int
		migMode     bool
		migProfiles = map[string]int{}
	)

	getLabel := func(m *dto.Metric, name string) string {
		for _, l := range m.GetLabel() {
			if l.GetName() == name {
				return l.GetValue()
			}
		}
		return ""
	}

	if mf, ok := families["DCGM_FI_DEV_COUNT"]; ok {
		gpuCount = int(mf.Metric[0].GetGauge().GetValue())
	}

	if mf, ok := families["DCGM_FI_DEV_NAME"]; ok {
		model = mf.Metric[0].GetLabel()[0].GetValue()
	}

	if mf, ok := families["DCGM_FI_DEV_FB_TOTAL"]; ok {
		vram = int(mf.Metric[0].GetGauge().GetValue())
	}

	// MIG detection
	if mf, ok := families["DCGM_FI_DEV_MIG_MODE"]; ok {
		if mf.Metric[0].GetGauge().GetValue() == 1 {
			migMode = true
		}
	}

	// MIG profiles
	if mf, ok := families["DCGM_FI_DEV_MIG_PROFILE"]; ok {
		for _, m := range mf.Metric {
			profile := getLabel(m, "profile")
			migProfiles[profile]++
		}
	}

	if gpuCount == 0 {
		return nil, fmt.Errorf("no gpus")
	}

	return &GPUInfo{
		NodeName:    node,
		GPUsPerNode: gpuCount,
		Model:       model,
		VRAMPerGPU:  vram,
		MIGEnabled:  migMode,
		MIGProfiles: migProfiles,
	}, nil
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
