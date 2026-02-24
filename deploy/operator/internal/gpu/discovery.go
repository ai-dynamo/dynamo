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
	"github.com/prometheus/common/model"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// NVIDIA GPU Feature Discovery (GFD) label keys
	LabelGPUCount   = "nvidia.com/gpu.count"
	LabelGPUProduct = "nvidia.com/gpu.product"
	LabelGPUMemory  = "nvidia.com/gpu.memory"
	// DCGM exporter label constants
	LabelApp                     = "app"
	LabelAppKubernetesName       = "app.kubernetes.io/name"
	LabelValueNvidiaDCGMExporter = "nvidia-dcgm-exporter"
	LabelValueDCGMExporter       = "dcgm-exporter"
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

// DiscoverGPUsFromDCGM discovers GPU information by scraping metrics directly
// from DCGM exporter pods running in the cluster.
//
// The function performs the following:
//
//  1. Returns cached GPU information if still valid.
//  2. Lists DCGM exporter pods across all namespaces using supported labels.
//  3. If no pods are found, attempts to enable DCGM via Helm.
//  4. Waits for DCGM exporter pods to become ready.
//  5. Scrapes each running pods metrics endpoint (http://<podIP>:9400/metrics).
//  6. Selects the "best" GPU node based on:
//     - Highest GPU count
//     - Highest VRAM per GPU (tie-breaker)
//  7. Caches the result for a short duration to avoid repeated scraping.
//
// Behavior Notes:
//
//   - Scrapes pods directly instead of using a Service ClusterIP to avoid
//     load-balancing ambiguity in multi-node clusters.
//   - If at least one pod is successfully scraped, partial failures are tolerated.
//   - If all pods fail to scrape, an aggregated error is returned.
//   - Assumes DCGM exporter runs as a DaemonSet (one pod per GPU node).
//   - Designed for homogeneous clusters; heterogeneous cluster aggregation
//     is not yet implemented.
//
// Returns:
//   - *GPUInfo for the selected node
//   - error if no GPU data can be retrieved
//
// TODO: Current implementation selects a single "best" GPU node (highest GPU count,
// tie-broken by VRAM). This works for homogeneous clusters where all GPU
// nodes are identical.
// For Heterogeneous GPU Support (mixed GPU models or capacities), this logic
// does not represent full cluster GPU inventory. Future improvements should
// aggregate and return GPU information for all nodes instead of selecting
// only one.
func DiscoverGPUsFromDCGM(ctx context.Context, k8sClient client.Client, cache *GPUDiscoveryCache) (*GPUInfo, error) {

	// Return cached result if still valid
	if cached, ok := cache.Get(); ok {
		return cached, nil
	}

	// List ALL pods (we'll filter manually to support OR label logic)
	podList := &corev1.PodList{}
	if err := k8sClient.List(ctx, podList); err != nil {
		return nil, fmt.Errorf("list pods: %w", err)
	}

	var dcgmPods []corev1.Pod

	// Filter pods matching ANY of the supported DCGM label combinations
	for _, pod := range podList.Items {

		labels := pod.GetLabels()

		if labels == nil {
			continue
		}

		// Supported label combinations:
		if labels[LabelApp] == LabelValueNvidiaDCGMExporter ||
			labels[LabelApp] == LabelValueDCGMExporter ||
			labels[LabelAppKubernetesName] == LabelValueDCGMExporter {

			dcgmPods = append(dcgmPods, pod)
		}
	}

	// If no pods found → attempt enablement
	if len(dcgmPods) == 0 {

		if err := EnsureDCGMEnabled(ctx); err != nil {
			return nil, fmt.Errorf("install dcgm via helm: %w", err)
		}

		if err := WaitForDCGMPods(ctx, k8sClient, 3*time.Minute); err != nil {
			return nil, err
		}

		// Re-list and re-filter
		if err := k8sClient.List(ctx, podList); err != nil {
			return nil, err
		}

		dcgmPods = nil

		for _, pod := range podList.Items {
			labels := pod.GetLabels()
			if labels == nil {
				continue
			}

			if labels[LabelApp] == LabelValueNvidiaDCGMExporter ||
				labels[LabelApp] == LabelValueDCGMExporter ||
				labels[LabelAppKubernetesName] == LabelValueDCGMExporter {

				dcgmPods = append(dcgmPods, pod)
			}
		}
	}

	if len(dcgmPods) == 0 {
		return nil, fmt.Errorf("no dcgm exporter pods found")
	}

	var bestNode *GPUInfo
	var scrapeErrors []error
	// Scrape each running pod individually
	for _, pod := range dcgmPods {

		if pod.Status.Phase != corev1.PodRunning {
			continue
		}

		if pod.Status.PodIP == "" {
			continue
		}

		endpoint := fmt.Sprintf("http://%s:9400/metrics", pod.Status.PodIP)

		info, err := scrapeMetricsEndpoint(ctx, endpoint)
		if err != nil {
			scrapeErrors = append(scrapeErrors, fmt.Errorf("pod %s (%s): %w", pod.Name, pod.Status.PodIP, err))
			continue
		}

		// Select best node:
		// 1. Highest GPU count
		// 2. Highest VRAM per GPU (tie-breaker)
		if bestNode == nil ||
			info.GPUsPerNode > bestNode.GPUsPerNode ||
			(info.GPUsPerNode == bestNode.GPUsPerNode &&
				info.VRAMPerGPU > bestNode.VRAMPerGPU) {

			bestNode = info
		}
	}

	if bestNode == nil {
		if len(scrapeErrors) > 0 {
			return nil, fmt.Errorf("failed to scrape any dcgm exporter pod: %v", scrapeErrors)
		}
		return nil, fmt.Errorf("no GPU metrics could be parsed from any dcgm pod")
	}

	// Cache result for 60 seconds
	cache.Set(bestNode, 60*time.Second)

	return bestNode, nil
}

// scrapeMetricsEndpoint retrieves and parses Prometheus metrics from a
// DCGM exporter pod endpoint.
//
// The function performs an HTTP GET request against the provided endpoint
// (expected format: http://<podIP>:9400/metrics), validates the response,
// and parses the Prometheus text exposition format into metric families.
//
// Parsed metric families are passed to parseMetrics to extract high-level
// GPU information.
//
// Returns:
//   - *GPUInfo derived from the parsed metrics
//   - error if the HTTP request fails, the response is non-200,
//     or metric parsing fails
//
// This function does not implement retries or fallback logic.
// Error handling and multi-pod aggregation are managed by the caller.
func scrapeMetricsEndpoint(ctx context.Context, endpoint string) (*GPUInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("create request for %s: %w", endpoint, err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET %s failed: %w", endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"metrics endpoint %s returned status %d",
			endpoint,
			resp.StatusCode,
		)
	}

	parser := expfmt.NewTextParser(model.UTF8Validation)
	metricFamilies, err := parser.TextToMetricFamilies(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("parse prometheus metrics: %w", err)
	}

	return parseMetrics(ctx, metricFamilies)
}

// parseMetrics extracts GPU information for a node from DCGM Prometheus metrics.
//
// It parses the provided Prometheus metric families exported by the NVIDIA
// DCGM exporter and derives high-level GPU inventory information for the node.
//
// The function performs the following:
//
//   - Detects the number of GPUs by counting unique "gpu" label values
//     from DCGM_FI_DEV_GPU_TEMP (used as a reliable per-GPU metric).
//
//   - Extracts the GPU model name from the "modelName" label.
//
//   - Calculates total VRAM per GPU using framebuffer metrics:
//     VRAM = FB_FREE + FB_USED + FB_RESERVED
//     (values are in MiB).
//
//   - Assumes MIG is disabled unless explicit MIG metrics are present
//     (not included in the provided DCGM metric set).
//
// Parameters:
//
//	ctx       - Context for logging and cancellation.
//	families  - Map of Prometheus metric families keyed by metric name.
//
// Returns:
//
//	*GPUInfo containing:
//	  - NodeName
//	  - GPUsPerNode
//	  - Model
//	  - VRAMPerGPU (MiB)
//	  - MIGEnabled: false because no MIG metrics were collected in the DCGM families
//	  - MIGProfiles: empty map; would contain MIG profile counts if MIG metrics were available
//	  - System (inferred from model)
//
// Returns an error if no GPUs can be detected from the metrics.
//
// Notes:
//   - This function relies on DCGM exporter metrics.
//   - If required metrics are missing, zero values may be returned.
//   - The implementation assumes homogeneous GPUs per node.
//   - For heterogeneous configurations, per-GPU parsing should be implemented.
func parseMetrics(ctx context.Context, families map[string]*dto.MetricFamily) (*GPUInfo, error) {
	logger := log.FromContext(ctx)

	getLabel := func(m *dto.Metric, name string) string {
		for _, l := range m.GetLabel() {
			if l.GetName() == name {
				return l.GetValue()
			}
		}
		return ""
	}

	// Track unique GPUs
	gpuSet := map[string]struct{}{}

	var model string
	var vram int
	var hostName string

	fbFree := map[string]float64{}
	fbUsed := map[string]float64{}
	fbReserved := map[string]float64{}

	// --- Detect GPUs + Model + Hostname ---
	if mf, ok := families["DCGM_FI_DEV_GPU_TEMP"]; ok {
		for _, m := range mf.Metric {
			gpuID := getLabel(m, "gpu")
			gpuSet[gpuID] = struct{}{}

			// Extract model from label
			if model == "" {
				model = getLabel(m, "modelName")
			}

			// Extract Hostname label
			if hostName == "" {
				hostName = getLabel(m, "Hostname")
			}
		}
	}

	// --- Collect framebuffer metrics ---
	if mf, ok := families["DCGM_FI_DEV_FB_FREE"]; ok {
		for _, m := range mf.Metric {
			gpuID := getLabel(m, "gpu")
			fbFree[gpuID] = m.GetGauge().GetValue()

			if hostName == "" {
				hostName = getLabel(m, "Hostname")
			}
		}
	}

	if mf, ok := families["DCGM_FI_DEV_FB_USED"]; ok {
		for _, m := range mf.Metric {
			gpuID := getLabel(m, "gpu")
			fbUsed[gpuID] = m.GetGauge().GetValue()

			if hostName == "" {
				hostName = getLabel(m, "Hostname")
			}
		}
	}

	if mf, ok := families["DCGM_FI_DEV_FB_RESERVED"]; ok {
		for _, m := range mf.Metric {
			gpuID := getLabel(m, "gpu")
			fbReserved[gpuID] = m.GetGauge().GetValue()

			if hostName == "" {
				hostName = getLabel(m, "Hostname")
			}
		}
	}

	// --- Calculate VRAM from first GPU ---
	for gpuID := range gpuSet {
		vram = int(fbFree[gpuID] + fbUsed[gpuID] + fbReserved[gpuID])
		break
	}

	gpuCount := len(gpuSet)

	if gpuCount == 0 {
		return nil, fmt.Errorf("no GPUs detected from DCGM metrics")
	}

	// --- Infer system from model ---
	system := InferHardwareSystem(model)

	logger.Info("Parsed GPU info",
		"node", hostName,
		"gpuCount", gpuCount,
		"model", model,
		"vramMiB", vram,
		"system", system,
	)

	return &GPUInfo{
		NodeName:    hostName,
		GPUsPerNode: gpuCount,
		Model:       model,
		VRAMPerGPU:  vram,
		MIGEnabled:  false,
		MIGProfiles: map[string]int{},
		System:      system, // populated from InferHardwareSystem
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
