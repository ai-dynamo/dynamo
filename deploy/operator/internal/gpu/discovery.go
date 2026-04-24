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
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	defaultDCGMEndpointTemplate = "http://{POD_IP}:9400/metrics"
	// NVIDIA GPU Feature Discovery (GFD) label keys
	LabelGPUCount   = "nvidia.com/gpu.count"
	LabelGPUProduct = "nvidia.com/gpu.product"
	LabelGPUMemory  = "nvidia.com/gpu.memory"
	// DCGM exporter label constants
	LabelApp                        = "app"
	LabelAppKubernetesName          = "app.kubernetes.io/name"
	LabelValueNvidiaDCGMExporter    = "nvidia-dcgm-exporter"
	LabelValueNvidiaNetworkOperator = "nvidia-network-operator"
	LabelValueDCGMExporter          = "dcgm-exporter"
	LabelValueGPUOperator           = "gpu-operator"
	GPUOperatorNamespace            = "gpu-operator"
	requestTimeout                  = 5 * time.Second
	dialTimeout                     = 3 * time.Second
	tlsHandshakeTimeout             = 3 * time.Second
	CloudProviderGCP                = "gcp"
	CloudProviderAWS                = "aws"
	CloudProviderAKS                = "aks"
	CloudProviderOther              = "other"
	CloudProviderUnknown            = "unknown"
)

// --- Normalization helpers ---
const (
	strDash  = "-"
	strSpace = " "
	strNone  = "none"
)

// --- Form factor tokens ---
const (
	tokenSXM       = "SXM"
	tokenHGX       = "HGX"
	tokenDGX       = "DGX"
	tokenPCIE      = "PCIE"
	formFactorSXM  = "sxm"
	formFactorPCIe = "pcie"
)

// --- GPU model tokens ---
const (
	tokenGB200  = "GB200"
	tokenB200   = "B200"
	tokenH200   = "H200"
	tokenH100   = "H100"
	tokenA100   = "A100"
	tokenL40S   = "L40S"
	tokenL40    = "L40"
	tokenL4     = "L4"
	tokenV100   = "V100"
	tokenT4     = "T4"
	tokenMI300  = "MI300"
	tokenMI250  = "MI250"
	tokenMI200  = "MI200"
	LabelNVLink = "nvlink"
)

// awsInstanceTypePrefixes matches known GPU/accelerator instance families on EKS. See: https://aws.amazon.com/ec2/instance-types/
var awsInstanceTypePrefixes = []string{
	"p3.", "p3dn.", "p4d.", "p4de.", "p5.", // GPU instances
	"g3.", "g4dn.", "g4ad.", "g5.", "g6.", // GPU instances
	"inf1.", "inf2.", // Inferentia
	"trn1.", "trn1n.", // Trainium
}

// gcpMachineSeries matches known GCP accelerator-optimised machine series on GKE. See: https://cloud.google.com/compute/docs/machine-resource
var gcpMachineSeries = []string{
	"a2-", // A100 GPU machines
	"a3-", // H100 GPU machines
	"g2-", // L4 GPU machines
}

type gpuRule struct {
	token     string
	sxmSKU    nvidiacomv1beta1.GPUSKUType
	pcieSKU   nvidiacomv1beta1.GPUSKUType
	singleSKU nvidiacomv1beta1.GPUSKUType // for GPUs without form factor variants
}

var gpuRules = []gpuRule{
	// Blackwell
	{token: tokenGB200, sxmSKU: nvidiacomv1beta1.GPUSKUTypeGB200SXM},
	{token: tokenB200, sxmSKU: nvidiacomv1beta1.GPUSKUTypeB200SXM},

	// Hopper
	{token: tokenH200, sxmSKU: nvidiacomv1beta1.GPUSKUTypeH200SXM},
	{token: tokenH100, sxmSKU: nvidiacomv1beta1.GPUSKUTypeH100SXM, pcieSKU: nvidiacomv1beta1.GPUSKUTypeH100PCIe},

	// Ampere
	{token: tokenA100, sxmSKU: nvidiacomv1beta1.GPUSKUTypeA100SXM, pcieSKU: nvidiacomv1beta1.GPUSKUTypeA100PCIe},

	// Ada
	{token: tokenL40S, singleSKU: nvidiacomv1beta1.GPUSKUTypeL40S},
	{token: tokenL40, singleSKU: nvidiacomv1beta1.GPUSKUTypeL40},
	{token: tokenL4, singleSKU: nvidiacomv1beta1.GPUSKUTypeL4},

	// Volta / Turing
	{token: tokenV100, sxmSKU: nvidiacomv1beta1.GPUSKUTypeV100SXM, pcieSKU: nvidiacomv1beta1.GPUSKUTypeV100PCIe},
	{token: tokenT4, singleSKU: nvidiacomv1beta1.GPUSKUTypeT4},

	// AMD
	{token: tokenMI300, singleSKU: nvidiacomv1beta1.GPUSKUTypeMI300},
	{token: tokenMI250, singleSKU: nvidiacomv1beta1.GPUSKUTypeMI200},
	{token: tokenMI200, singleSKU: nvidiacomv1beta1.GPUSKUTypeMI200},
}

// GPUInfo contains discovered GPU configuration from cluster nodes
type GPUInfo struct {
	NodeName         string                      // Name of the node with this GPU configuration
	GPUsPerNode      int                         // Maximum GPUs per node found in the cluster
	NodesWithGPUs    int                         // Number of nodes that have GPUs
	Model            string                      // GPU product name (e.g., "H100-SXM5-80GB")
	VRAMPerGPU       int                         // VRAM in MiB per GPU
	System           nvidiacomv1beta1.GPUSKUType // AIC hardware system identifier (e.g., "h100_sxm", "h200_sxm"), empty if unknown
	MIGEnabled       bool                        // True if MIG is enabled (inferred from model or additional labels, not implemented in this version)
	MIGProfiles      map[string]int              // Optional: map of MIG profile name to count (requires additional label parsing, not implemented in this version)
	CloudProvider    string                      // aws | gcp | aks | other | unknown
	RDMAEnabled      bool                        // Indicates whether RDMA is enabled for this node (e.g., via InfiniBand, RoCE, or similar high-speed networking)
	RDMAType         string                      // Type of RDMA transport detected (e.g., "infiniband", "roce", "rdma", "sriov", or "none")
	Interconnect     string                      // Primary GPU-to-GPU interconnect technology used within the node (e.g., "nvlink" for high-bandwidth links or "pcie" for standard bus-based communication)
	InterconnectTier string                      // Qualitative or platform-specific classification of the interconnect (e.g., NVLink generation, topology tier, or vendor-defined performance level)
	NVLinkLinks      int                         // Number of NVLink connections per GPU (0 if NVLink is not present or interconnect is PCIe-only)
}

type ScrapeMetricsFunc func(ctx context.Context, endpoint string) (*GPUInfo, error)

type gpuCacheEntry struct {
	value     *GPUInfo
	expiresAt time.Time
}

// GPUDiscoveryCache caches discovery results keyed by SKU filter.
// Bounded by the GPUSKUType enum (≤7 values incl. empty for unfiltered).
type GPUDiscoveryCache struct {
	mu      sync.RWMutex
	entries map[nvidiacomv1beta1.GPUSKUType]gpuCacheEntry
}

type GPUDiscovery struct {
	Scraper ScrapeMetricsFunc
}

func NewGPUDiscovery(scraper ScrapeMetricsFunc) *GPUDiscovery {
	return &GPUDiscovery{
		Scraper: scraper,
	}
}

// NewGPUDiscoveryCache creates a new GPUDiscoveryCache instance.
//
// The cache stores discovered GPUInfo values keyed by SKU filter with an
// expiration time. It is safe for concurrent use and is intended to reduce
// repeated DCGM scraping during reconciliation loops.
func NewGPUDiscoveryCache() *GPUDiscoveryCache {
	return &GPUDiscoveryCache{
		entries: make(map[nvidiacomv1beta1.GPUSKUType]gpuCacheEntry),
	}
}

// Get returns the cached GPUInfo for the given SKU filter if it exists and
// has not expired.
//
// The boolean return value indicates whether a valid cached value was found.
// If the cache is empty or expired, it returns (nil, false).
//
// This method is safe for concurrent use.
func (c *GPUDiscoveryCache) Get(sku nvidiacomv1beta1.GPUSKUType) (*GPUInfo, bool) {
	c.mu.RLock()
	e, ok := c.entries[sku]
	c.mu.RUnlock()
	if ok && time.Now().Before(e.expiresAt) && e.value != nil {
		return e.value, true
	}
	return nil, false
}

// Set stores the provided GPUInfo in the cache with the given TTL (time-to-live).
//
// The cached value will be considered valid until the TTL duration elapses.
// After expiration, Get will return (nil, false) until a new value is set.
//
// This method is safe for concurrent use.
func (c *GPUDiscoveryCache) Set(sku nvidiacomv1beta1.GPUSKUType, info *GPUInfo, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[sku] = gpuCacheEntry{value: info, expiresAt: time.Now().Add(ttl)}
}

// DiscoverGPUsFromDCGM is a convenience wrapper that calls
// DiscoverGPUsFromDCGMFiltered with no SKU filter.
// See DiscoverGPUsFromDCGMFiltered for full documentation.
func (g *GPUDiscovery) DiscoverGPUsFromDCGM(ctx context.Context, k8sClient client.Reader, cache *GPUDiscoveryCache) (*GPUInfo, error) {
	return g.DiscoverGPUsFromDCGMFiltered(ctx, k8sClient, cache, "")
}

// DiscoverGPUsFromDCGMFiltered discovers GPU information by scraping metrics
// directly from DCGM exporter pods running in the cluster.
//
// When filterSKU is non-empty, only nodes whose inferred SKU matches are
// considered. When empty, the best node is selected first (highest GPU count,
// then VRAM) and then only nodes with the same SKU are counted.
//
// The function performs the following:
//
//  1. Returns cached GPU information if still valid (keyed by filterSKU).
//  2. Lists DCGM exporter pods across all namespaces using supported labels.
//  3. If no pods are found, attempts to find if GPU operator is installed and DCGM is enabled via Helm.
//  4. Warns user appropriately.
//  5. Scrapes each running pods metrics endpoint (http://<podIP>:9400/metrics).
//  6. Selects the "best" GPU node (filtered by SKU when set) based on:
//     - Highest GPU count
//     - Highest VRAM per GPU (tie-breaker)
//  7. Counts only nodes matching the selected SKU for NodesWithGPUs.
//  8. Caches the result per SKU for a short duration to avoid repeated scraping.
//
// Behavior Notes:
//
//   - Scrapes pods directly instead of using a Service ClusterIP to avoid
//     load-balancing ambiguity in multi-node clusters.
//   - If at least one pod is successfully scraped, partial failures are tolerated.
//   - If all pods fail to scrape, an aggregated error is returned.
//   - Assumes DCGM exporter runs as a DaemonSet (one pod per GPU node).
//
// Returns:
//   - *GPUInfo for the selected node
//   - error if no GPU data can be retrieved
func (g *GPUDiscovery) DiscoverGPUsFromDCGMFiltered(ctx context.Context, k8sClient client.Reader, cache *GPUDiscoveryCache, filterSKU nvidiacomv1beta1.GPUSKUType) (*GPUInfo, error) {
	if cache != nil {
		if cached, ok := cache.Get(filterSKU); ok {
			return cached, nil
		}
	}

	// List DCGM exporter pods
	dcgmPods, err := listDCGMExporterPods(ctx, k8sClient)
	if err != nil && !strings.Contains(err.Error(), "no DCGM exporter pods found") {
		return nil, fmt.Errorf("listing DCGM exporter pods failed: %w", err)
	}
	// If no pods found
	if len(dcgmPods) == 0 {
		gpuPods, err := listGPUOperatorRunningPods(ctx, k8sClient)
		if len(gpuPods) > 0 {
			return nil, fmt.Errorf("DCGM is not enabled in the GPU Operator (check GPU Operator configuration and permissions)")
		}
		return nil, err
	}

	// Scrape each running pod and collect per-node GPU info.
	type nodeInfo struct {
		info     *GPUInfo
		sku      nvidiacomv1beta1.GPUSKUType
		nodeName string
	}
	allNodes := make([]nodeInfo, 0, len(dcgmPods))
	var scrapeErrors []error

	for _, pod := range dcgmPods {
		if pod.Status.Phase != corev1.PodRunning || pod.Status.PodIP == "" {
			continue
		}
		endpoint := buildDCGMEndpoint(pod.Status.PodIP)
		info, err := g.Scraper(ctx, endpoint)
		if err != nil {
			scrapeErrors = append(scrapeErrors, fmt.Errorf("pod %s (%s): %w", pod.Name, pod.Status.PodIP, err))
			continue
		}

		allNodes = append(allNodes, nodeInfo{info: info, sku: InferHardwareSystem(info.Model), nodeName: pod.Spec.NodeName})
	}

	if len(allNodes) == 0 {
		if len(scrapeErrors) > 0 {
			return nil, fmt.Errorf("failed to scrape any DCGM exporter pod: %v", scrapeErrors)
		}
		return nil, fmt.Errorf("no GPU metrics could be parsed from any DCGM pod")
	}

	// Select best node (only from matching SKU when filtered).
	var bestNode *GPUInfo
	var bestSKU nvidiacomv1beta1.GPUSKUType
	for _, n := range allNodes {
		if filterSKU != "" && n.sku != filterSKU {
			continue
		}
		if bestNode == nil ||
			n.info.GPUsPerNode > bestNode.GPUsPerNode ||
			(n.info.GPUsPerNode == bestNode.GPUsPerNode &&
				n.info.VRAMPerGPU > bestNode.VRAMPerGPU) {
			bestNode = n.info
			bestSKU = n.sku
		}
	}

	if bestNode == nil {
		if filterSKU != "" {
			return nil, fmt.Errorf("no GPU nodes matching SKU %q found", filterSKU)
		}
		return nil, fmt.Errorf("no GPU metrics could be parsed from any DCGM pod")
	}

	// Count only nodes with the same SKU as the selected best node,
	// and detect RDMA on matching nodes only.
	nodesWithGPUs := 0
	var rdmaDetected bool
	var rdmaType string
	for _, n := range allNodes {
		if n.sku != bestSKU {
			continue
		}
		nodesWithGPUs++
		if !rdmaDetected {
			rdma, rType := detectRDMAFromNode(ctx, k8sClient, n.nodeName)
			if rdma {
				rdmaDetected = true
				rdmaType = rType
			}
		}
	}

	// Detect InfiniBand presence
	ib := detectIBPods(ctx, k8sClient)
	if ib {
		rdmaType = "infiniband"
		rdmaDetected = true
	}

	cloudProvider, err := GetCloudProviderInfo(ctx, k8sClient)
	if err != nil {
		cloudProvider = CloudProviderUnknown
	}
	bestNode.System = bestSKU
	bestNode.CloudProvider = cloudProvider
	bestNode.NodesWithGPUs = nodesWithGPUs
	bestNode.RDMAEnabled = rdmaDetected
	bestNode.RDMAType = rdmaType

	if cache != nil {
		cache.Set(filterSKU, bestNode, 60*time.Second)
	}
	return bestNode, nil
}
func buildDCGMEndpoint(podIP string) string {
	template := os.Getenv("DCGM_METRICS_ENDPOINT_TEMPLATE")
	if template == "" {
		template = defaultDCGMEndpointTemplate
	}
	return strings.ReplaceAll(template, "{POD_IP}", podIP)
}
func listDCGMExporterPods(ctx context.Context, k8sClient client.Reader) ([]corev1.Pod, error) {
	var result []corev1.Pod
	seen := make(map[string]struct{})
	selectors := []client.MatchingLabels{
		{LabelApp: LabelValueNvidiaDCGMExporter},
		{LabelApp: LabelValueDCGMExporter},
		{LabelAppKubernetesName: LabelValueDCGMExporter},
	}
	var lastErr error
	for _, selector := range selectors {
		podList := &corev1.PodList{}
		err := k8sClient.List(ctx, podList, selector)
		if err != nil {
			lastErr = fmt.Errorf("list pods: %w", err)
			continue
		}
		for _, pod := range podList.Items {
			key := pod.Namespace + "/" + pod.Name
			if _, exists := seen[key]; !exists {
				seen[key] = struct{}{}
				result = append(result, pod)
			}
		}
	}
	if len(result) > 0 {
		return result, nil
	}
	if lastErr != nil {
		return nil, lastErr
	}
	return nil, fmt.Errorf("no DCGM exporter pods found")
}

// listGPUOperatorRunningPods lists GPU Operator pods in the given namespace
// and returns only those that are in Running phase.
//
// It uses common GPU Operator label selectors and deduplicates results
// across selectors. If no running pods are found, an error is returned.
func listGPUOperatorRunningPods(ctx context.Context, k8sClient client.Reader) ([]corev1.Pod, error) {
	var result []corev1.Pod
	seen := make(map[string]struct{})
	selectors := []client.MatchingLabels{
		{LabelApp: LabelValueGPUOperator},
		{LabelAppKubernetesName: LabelValueGPUOperator},
	}
	var lastErr error
	for _, selector := range selectors {
		podList := &corev1.PodList{}
		err := k8sClient.List(
			ctx,
			podList,
			client.InNamespace(GPUOperatorNamespace),
			selector,
		)
		if err != nil {
			lastErr = fmt.Errorf("list gpu operator pods: %w", err)
			continue
		}
		for _, pod := range podList.Items {
			if pod.Status.Phase != corev1.PodRunning {
				continue
			}
			key := pod.Namespace + "/" + pod.Name
			if _, exists := seen[key]; !exists {
				seen[key] = struct{}{}
				result = append(result, pod)
			}
		}
	}
	if len(result) > 0 {
		return result, nil
	}
	if lastErr != nil {
		return nil, lastErr
	}
	return nil, fmt.Errorf(
		"gpu operator is not installed %s",
		GPUOperatorNamespace,
	)
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
func ScrapeMetricsEndpoint(ctx context.Context, endpoint string) (*GPUInfo, error) {
	// Set a timeout for the request
	ctx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()
	// Create a custom HTTP client with transport-level timeouts
	client := &http.Client{
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   dialTimeout,      // Dial timeout
				KeepAlive: 30 * time.Second, // Keep-alive for connections
			}).DialContext,
			TLSHandshakeTimeout: tlsHandshakeTimeout, // TLS handshake timeout
		},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("create request for %s: %w", endpoint, err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP GET %s failed: %w", endpoint, err)
	}
	defer func() {
		if cerr := resp.Body.Close(); cerr != nil {
			// best-effort: can't return an error from defer; log it
			log.FromContext(ctx).V(1).Info("failed to close response body", "err", cerr)
		}
	}()
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

// parseMetrics extracts GPU information and interconnect type for a node from DCGM Prometheus metrics.
//
// It parses the provided Prometheus metric families exported by the NVIDIA
// DCGM exporter and derives high-level GPU inventory and interconnect information for the node.
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
//   - Determines the interconnect type (PCIe or NVLink) from the
//     DCGM_FI_DEV_NVLINK_LINK_COUNT metric. If NVLink links are present,
//     interconnect is set to "nvlink", otherwise defaults to "pcie".
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
//	  - Interconnect: "pcie" or "nvlink" depending on detected NVLink links
//
// Returns an error if no GPUs can be detected from the metrics.
//
// Notes:
//   - This function relies on DCGM exporter metrics.
//   - If required metrics are missing, zero values may be returned.
//   - Interconnect detection is based on NVLink link count; other interconnects are not currently detected.
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
	var nvlinkDetected bool
	var nvlinkLinks int
	fbFree := map[string]float64{}
	fbUsed := map[string]float64{}
	fbReserved := map[string]float64{}
	// --- Detect GPUs + Model + Hostname ---
	if mf, ok := families["DCGM_FI_DEV_GPU_TEMP"]; ok {
		for _, m := range mf.Metric {
			gpuID := getLabel(m, "gpu")
			if gpuID == "" {
				continue
			}
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
			if gpuID == "" {
				continue
			}
			fbFree[gpuID] = m.GetGauge().GetValue()
			if hostName == "" {
				hostName = getLabel(m, "Hostname")
			}
		}
	}
	if mf, ok := families["DCGM_FI_DEV_FB_USED"]; ok {
		for _, m := range mf.Metric {
			gpuID := getLabel(m, "gpu")
			if gpuID == "" {
				continue
			}
			fbUsed[gpuID] = m.GetGauge().GetValue()
			if hostName == "" {
				hostName = getLabel(m, "Hostname")
			}
		}
	}
	if mf, ok := families["DCGM_FI_DEV_FB_RESERVED"]; ok {
		for _, m := range mf.Metric {
			gpuID := getLabel(m, "gpu")
			if gpuID == "" {
				continue
			}
			fbReserved[gpuID] = m.GetGauge().GetValue()
			if hostName == "" {
				hostName = getLabel(m, "Hostname")
			}
		}
	}
	if mf, ok := families["DCGM_FI_DEV_NVLINK_LINK_COUNT"]; ok {
		for _, m := range mf.Metric {
			val := int(m.GetGauge().GetValue())
			if val > 0 {
				nvlinkDetected = true
				nvlinkLinks = val
				break
			}
		}
	}
	// --- Determine interconnect type ---
	interconnect := "pcie"
	interconnectDetail := strNone
	if nvlinkDetected {
		switch {
		case nvlinkLinks >= 12:
			interconnect = LabelNVLink
			interconnectDetail = "full-mesh" // HGX / DGX class
		case nvlinkLinks >= 6:
			interconnect = LabelNVLink
			interconnectDetail = "high"
		default:
			interconnect = LabelNVLink
			interconnectDetail = "partial"
		}
	}
	// --- Calculate Max VRAM
	for gpuID := range gpuSet {
		total := int(fbFree[gpuID] + fbUsed[gpuID] + fbReserved[gpuID])
		if total > vram {
			vram = total
		}
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
		"interconnect", interconnect,
		"interconnectDetail", interconnectDetail,
		"nvlinkLinks", nvlinkLinks,
	)
	return &GPUInfo{
		NodeName:         hostName,
		GPUsPerNode:      gpuCount,
		Model:            model,
		VRAMPerGPU:       vram,
		MIGEnabled:       false,
		MIGProfiles:      map[string]int{},
		System:           system, // populated from InferHardwareSystem
		Interconnect:     interconnect,
		InterconnectTier: interconnectDetail,
		NVLinkLinks:      nvlinkLinks,
	}, nil
}

// DiscoverGPUs queries Kubernetes nodes to determine GPU configuration.
// It extracts GPU information from NVIDIA GPU Feature Discovery (GFD) labels
// and returns aggregated GPU info, preferring nodes with higher GPU count,
// then higher VRAM if counts are equal.
//
// This function requires cluster-wide node read permissions and expects nodes
// to have GFD labels. If no nodes with GPU labels are found, it returns an error.
func DiscoverGPUs(ctx context.Context, k8sClient client.Reader) (*GPUInfo, error) {
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
	bestGPUInfo.NodesWithGPUs = nodesWithGPUs
	logger.Info("GPU discovery completed",
		"gpusPerNode", bestGPUInfo.GPUsPerNode,
		"nodesWithGPUs", bestGPUInfo.NodesWithGPUs,
		"totalGpus", bestGPUInfo.GPUsPerNode*bestGPUInfo.NodesWithGPUs,
		"model", bestGPUInfo.Model,
		"vram", bestGPUInfo.VRAMPerGPU,
		"system", bestGPUInfo.System)
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

// InferHardwareSystem attempts to infer a normalized GPU SKU type from a
// free-form product string (e.g. "NVIDIA H100 SXM", "A100-PCIE").
//
// The function performs three main steps:
//  1. Normalize the input string to a consistent format.
//  2. Detect the GPU form factor (SXM vs PCIe).
//  3. Match the normalized string against known GPU tokens and return
//     the corresponding SKU type.
//
// Matching is based on substring checks and is tolerant of variations
// in formatting (case, spaces, dashes). If no known GPU is detected,
// an empty SKU type is returned.
// Limitations:
//   - Cannot distinguish SXM vs. PCIe variants from labels alone (assumes SXM for datacenter GPUs)
//   - New GPU models require code updates (gracefully returns empty string)
//   - Non-standard SKU names may not match
//
// Users can manually override the system in their profiling config (hardware.system)
// if auto-detection is incorrect or unavailable.
func InferHardwareSystem(gpuProduct string) nvidiacomv1beta1.GPUSKUType {
	if gpuProduct == "" {
		return ""
	}

	normalized := normalize(gpuProduct)
	formFactor := detectFormFactor(normalized)

	for _, rule := range gpuRules {
		if strings.Contains(normalized, rule.token) {
			if rule.singleSKU != "" {
				return rule.singleSKU
			}
			if formFactor == formFactorSXM && rule.sxmSKU != "" {
				return rule.sxmSKU
			}
			if rule.pcieSKU != "" {
				return rule.pcieSKU
			}
			// Token matched but no form factor indicator was present in the string
			// (e.g. "NVIDIA H200" from DCGM has no SXM/HGX/DGX suffix). If the GPU
			// has no PCIe variant it must be SXM-only (H200, B200, GB200).
			if rule.sxmSKU != "" {
				return rule.sxmSKU
			}
		}
	}

	return ""
}

// normalize standardizes a GPU product string to simplify matching.
//
// It converts the string to uppercase and removes common separators
// such as spaces and dashes. This allows consistent substring matching
// regardless of how the input is formatted (e.g. "H100-SXM",
// "h100 sxm", and "H100SXM" all normalize to the same value).
func normalize(input string) string {
	s := strings.ToUpper(strings.ReplaceAll(input, strDash, strSpace))
	return strings.ReplaceAll(s, " ", "")
}

// detectFormFactor determines the GPU form factor (e.g. SXM or PCIe)
// from a normalized product string.
//
// The detection is based on the presence of known substrings such as
// "SXM", "HGX", or "DGX" for SXM-based systems, and "PCIE" for PCIe.
// If no explicit indicator is found, PCIe is used as the default since
// it is the more common and safer assumption.
func detectFormFactor(normalized string) string {
	switch {
	case strings.Contains(normalized, tokenSXM),
		strings.Contains(normalized, tokenHGX),
		strings.Contains(normalized, tokenDGX):
		return formFactorSXM
	case strings.Contains(normalized, tokenPCIE):
		return formFactorPCIe
	default:
		return formFactorPCIe
	}
}

// GetCloudProviderInfo attempts to infer the cloud provider of the Kubernetes cluster.
//
// The function inspects the first node in the cluster (assumes homogeneous node setup)
// and uses a combination of ProviderID and node labels to detect the provider.
//
// Detection logic:
//   - Primary detection uses node.Spec.ProviderID:
//   - "azure" → AKS
//   - "aws"   → AWS
//   - "gce"   → GCP
//   - Secondary detection uses node labels and instance type prefixes:
//   - AKS: "kubernetes.azure.com/cluster" label or instance type starting with "standard_"
//   - AWS: "eks.amazonaws.com/nodegroup" label or known AWS instance type prefix
//   - GCP: "cloud.google.com/gke-nodepool" label or known GCP machine series prefix
//   - If none match, returns "other".
//
// Parameters:
//   - ctx: Context for logging, cancellation, or timeout.
//   - k8sClient: Kubernetes client for reading Node objects.
//
// Returns:
//   - A string identifying the cloud provider ("aks", "aws", "gcp", "other", or "unknown").
//   - An error if no nodes are found or listing fails.
func GetCloudProviderInfo(ctx context.Context, k8sClient client.Reader) (string, error) {
	var nodeList corev1.NodeList
	if err := k8sClient.List(ctx, &nodeList); err != nil {
		return CloudProviderUnknown, fmt.Errorf("failed to list nodes: %w", err)
	}
	if len(nodeList.Items) == 0 {
		return CloudProviderUnknown, fmt.Errorf("no nodes found in cluster")
	}
	// Use first node as representative (assumes homogeneous control plane)
	node := nodeList.Items[0]
	providerID := strings.ToLower(node.Spec.ProviderID)
	labels := node.Labels
	instanceType := strings.ToLower(labels["node.kubernetes.io/instance-type"])
	// ---- Primary Detection: providerID ----
	switch {
	case strings.Contains(providerID, "azure"):
		return CloudProviderAKS, nil
	case strings.Contains(providerID, "aws"):
		return CloudProviderAWS, nil
	case strings.Contains(providerID, "gce"):
		return CloudProviderGCP, nil
	}
	// ---- Secondary Detection: Node Labels ----
	// AKS labels
	if _, ok := labels["kubernetes.azure.com/cluster"]; ok {
		return CloudProviderAKS, nil
	}
	if strings.Contains(instanceType, "standard_") {
		return CloudProviderAKS, nil
	}
	// EKS labels
	if _, ok := labels["eks.amazonaws.com/nodegroup"]; ok {
		return CloudProviderAWS, nil
	}
	if isAWSInstanceType(instanceType) {
		return CloudProviderAWS, nil
	}
	// GKE labels
	if _, ok := labels["cloud.google.com/gke-nodepool"]; ok {
		return CloudProviderGCP, nil
	}
	if isGCPInstanceType(instanceType) {
		return CloudProviderGCP, nil
	}
	return "other", nil
}

// isGCPInstanceType checks whether a given instance type string matches known GCP machine series.
//
// Parameters:
//   - instanceType: string representing the node's instance type (lowercased).
//
// Returns:
//   - true if the instance type belongs to a GCP machine series prefix.
func isGCPInstanceType(instanceType string) bool {
	for _, prefix := range gcpMachineSeries {
		if strings.HasPrefix(instanceType, prefix) {
			return true
		}
	}
	return false
}

// isAWSInstanceType checks whether a given instance type string matches known AWS instance type prefixes.
//
// Parameters:
//   - instanceType: string representing the node's instance type (lowercased).
//
// Returns:
//   - true if the instance type belongs to an AWS instance type prefix.
func isAWSInstanceType(instanceType string) bool {
	for _, prefix := range awsInstanceTypePrefixes {
		if strings.HasPrefix(instanceType, prefix) {
			return true
		}
	}
	return false
}

// detectRDMAFromNode inspects a single node for RDMA or SR-IOV network capability.
//
// Detection logic:
//   - Checks node labels:
//   - "nvidia.com/rdma.present" = "true" → RDMA detected
//   - "feature.node.kubernetes.io/network-sriov.capable" = "true" → SR-IOV detected
//
// Parameters:
//   - ctx: Context for logging or cancellation.
//   - k8sClient: Kubernetes client for reading Node objects.
//   - nodeName: Name of the node to inspect.
//
// Returns:
//   - bool indicating whether RDMA/SR-IOV is present.
//   - string representing the type ("rdma", "sriov", or "none").
func detectRDMAFromNode(ctx context.Context, k8sClient client.Reader, nodeName string) (bool, string) {
	node := &corev1.Node{}
	if err := k8sClient.Get(ctx, types.NamespacedName{Name: nodeName}, node); err != nil {
		return false, strNone
	}
	labels := node.Labels
	if labels["nvidia.com/rdma.present"] == "true" {
		return true, "rdma"
	}
	if labels["feature.node.kubernetes.io/network-sriov.capable"] == "true" {
		return true, "sriov"
	}
	return false, strNone
}

// detectIBPods checks if there are any RDMA or InfiniBand-related pods deployed
// in the "nvidia-network-operator" namespace.
//
// Detection logic:
//   - Lists pods in "nvidia-network-operator" namespace.
//   - If any pod name contains "rdma", returns true.
//
// Parameters:
//   - ctx: Context for logging or cancellation.
//   - k8sClient: Kubernetes client for listing pods.
//
// Returns:
//   - true if any RDMA/IB pods are found, false otherwise.
func detectIBPods(ctx context.Context, k8sClient client.Reader) bool {
	podList := &corev1.PodList{}
	if err := k8sClient.List(ctx, podList, client.InNamespace(LabelValueNvidiaNetworkOperator)); err != nil {
		return false
	}
	for _, p := range podList.Items {
		if strings.Contains(p.Name, "rdma") {
			return true
		}
	}
	return false
}
