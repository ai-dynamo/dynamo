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
	"math"
	"strings"

	dto "github.com/prometheus/client_model/go"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func hasIntelMetricFamilies(families map[string]*dto.MetricFamily) bool {
	_, ok := getMetricFamily(families, "hw_gpu_info", "hw.gpu.info")
	return ok
}

func parseMetrics(ctx context.Context, families map[string]*dto.MetricFamily) (*GPUInfo, error) {
	info, err := parseDCGMAcceleratorMetrics(ctx, families)
	if err != nil {
		return nil, err
	}
	return info.toGPUInfo(), nil
}

func parseDCGMAcceleratorMetrics(ctx context.Context, families map[string]*dto.MetricFamily) (*DiscoveredAcceleratorInfo, error) {
	logger := log.FromContext(ctx)
	getLabel := func(m *dto.Metric, name string) string {
		for _, l := range m.GetLabel() {
			if l.GetName() == name {
				return l.GetValue()
			}
		}
		return ""
	}

	gpuSet := map[string]struct{}{}
	var model string
	var vram int
	var hostName string
	var nvlinkDetected bool
	var nvlinkLinks int
	fbFree := map[string]float64{}
	fbUsed := map[string]float64{}
	fbReserved := map[string]float64{}

	if mf, ok := families["DCGM_FI_DEV_GPU_TEMP"]; ok {
		for _, m := range mf.Metric {
			gpuID := getLabel(m, "gpu")
			if gpuID == "" {
				continue
			}
			gpuSet[gpuID] = struct{}{}
			if model == "" {
				model = getLabel(m, "modelName")
			}
			if hostName == "" {
				hostName = getLabel(m, "Hostname")
			}
		}
	}

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

	interconnect := formFactorPCIe
	interconnectDetail := strNone
	if nvlinkDetected {
		switch {
		case nvlinkLinks >= 12:
			interconnect = LabelNVLink
			interconnectDetail = "full-mesh"
		case nvlinkLinks >= 6:
			interconnect = LabelNVLink
			interconnectDetail = "high"
		default:
			interconnect = LabelNVLink
			interconnectDetail = "partial"
		}
	}

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

	return &DiscoveredAcceleratorInfo{
		NodeName:                hostName,
		AcceleratorsPerNode:     gpuCount,
		Model:                   model,
		MemoryPerAcceleratorMiB: vram,
		SKU:                     system,
		MIGEnabled:              false,
		MIGProfiles:             map[string]int{},
		Interconnect:            interconnect,
		InterconnectTier:        interconnectDetail,
		NVLinkLinks:             nvlinkLinks,
	}, nil
}

func parseIntelMetrics(ctx context.Context, families map[string]*dto.MetricFamily) (*GPUInfo, error) {
	info, err := parseIntelAcceleratorMetrics(ctx, families)
	if err != nil {
		return nil, err
	}
	return info.toGPUInfo(), nil
}

func parseIntelAcceleratorMetrics(ctx context.Context, families map[string]*dto.MetricFamily) (*DiscoveredAcceleratorInfo, error) {
	logger := log.FromContext(ctx)
	infoFamily, ok := getMetricFamily(families, "hw_gpu_info", "hw.gpu.info")
	if !ok || len(infoFamily.Metric) == 0 {
		return nil, fmt.Errorf("no XPU devices detected from XPUMD metrics")
	}

	nodeName := ""
	model := ""
	pciDeviceID := ""
	maxVRAMMiB := 0
	deviceIDs := make(map[string]struct{})
	memoryByDevice := map[string]int{}
	if mf, ok := getMetricFamily(families, "hw_memory_size_bytes", "hw_memory_size", "hw.memory.size"); ok {
		for _, m := range mf.Metric {
			deviceID := getMetricLabel(m, "device_id", "hw_id", "hw.id", "pci_bdf", "pci.bdf")
			if deviceID == "" {
				continue
			}
			location := strings.ToLower(strings.TrimSpace(getMetricLabel(m, "hw_memory_location", "hw.memory.location")))
			if location != "" && location != "device" {
				continue
			}
			value, ok := getMetricFloatValue(m)
			if !ok {
				continue
			}
			memoryByDevice[deviceID] = max(memoryByDevice[deviceID], int(math.Round(value/(1024*1024))))
			if nodeName == "" {
				nodeName = getMetricLabel(m, "node_name")
			}
		}
	}

	for _, m := range infoFamily.Metric {
		deviceID := getMetricLabel(m, "device_id", "hw_id", "hw.id", "pci_bdf", "pci.bdf")
		if deviceID == "" {
			continue
		}
		deviceIDs[deviceID] = struct{}{}
		if model == "" {
			model = getMetricLabel(m, "device_name", "hw_model", "hw.model", "hw_name", "hw.name")
		}
		if pciDeviceID == "" {
			pciDeviceID = getMetricLabel(m, "pci_device_id", "pci.device_id")
		}
		if nodeName == "" {
			nodeName = getMetricLabel(m, "node_name")
		}
		if total, ok := memoryByDevice[deviceID]; ok && total > maxVRAMMiB {
			maxVRAMMiB = total
		}
	}

	deviceCount := len(deviceIDs)
	if deviceCount == 0 {
		return nil, fmt.Errorf("no XPU devices detected from XPUMD metrics")
	}

	system := inferIntelHardwareSystem(model, pciDeviceID, maxVRAMMiB)
	logger.Info("Parsed Intel XPU info",
		"node", nodeName,
		"gpuCount", deviceCount,
		"model", model,
		"pciDeviceID", pciDeviceID,
		"vramMiB", maxVRAMMiB,
		"system", system,
	)

	return &DiscoveredAcceleratorInfo{
		NodeName:                nodeName,
		AcceleratorsPerNode:     deviceCount,
		Model:                   model,
		PCIDeviceID:             pciDeviceID,
		MemoryPerAcceleratorMiB: maxVRAMMiB,
		MIGEnabled:              false,
		MIGProfiles:             map[string]int{},
		SKU:                     system,
		Interconnect:            formFactorPCIe,
		InterconnectTier:        strNone,
	}, nil
}

func getMetricFamily(families map[string]*dto.MetricFamily, names ...string) (*dto.MetricFamily, bool) {
	for _, name := range names {
		if mf, ok := families[name]; ok {
			return mf, true
		}
	}
	return nil, false
}

func getMetricLabel(m *dto.Metric, names ...string) string {
	for _, name := range names {
		for _, l := range m.GetLabel() {
			if l.GetName() == name {
				return l.GetValue()
			}
		}
	}
	return ""
}

func getMetricFloatValue(m *dto.Metric) (float64, bool) {
	switch {
	case m.GetGauge() != nil:
		return m.GetGauge().GetValue(), true
	case m.GetCounter() != nil:
		return m.GetCounter().GetValue(), true
	case m.GetUntyped() != nil:
		return m.GetUntyped().GetValue(), true
	default:
		return 0, false
	}
}
