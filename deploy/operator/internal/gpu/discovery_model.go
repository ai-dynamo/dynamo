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

import nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"

// DiscoveredAcceleratorInfo is the normalized internal discovery model shared by
// source-specific parsers and cluster-level selection logic.
type DiscoveredAcceleratorInfo struct {
	NodeName                string
	AcceleratorsPerNode     int
	NodesWithAccelerators   int
	Model                   string
	PCIDeviceID             string
	MemoryPerAcceleratorMiB int
	SKU                     nvidiacomv1beta1.GPUSKUType
	MIGEnabled              bool
	MIGProfiles             map[string]int
	CloudProvider           string
	RDMAEnabled             bool
	RDMAType                string
	Interconnect            string
	InterconnectTier        string
	NVLinkLinks             int
}

func discoveredAcceleratorInfoFromGPUInfo(info *GPUInfo) *DiscoveredAcceleratorInfo {
	if info == nil {
		return nil
	}

	return &DiscoveredAcceleratorInfo{
		NodeName:                info.NodeName,
		AcceleratorsPerNode:     info.GPUsPerNode,
		NodesWithAccelerators:   info.NodesWithGPUs,
		Model:                   info.Model,
		MemoryPerAcceleratorMiB: info.VRAMPerGPU,
		SKU:                     info.System,
		MIGEnabled:              info.MIGEnabled,
		MIGProfiles:             cloneMIGProfiles(info.MIGProfiles),
		CloudProvider:           info.CloudProvider,
		RDMAEnabled:             info.RDMAEnabled,
		RDMAType:                info.RDMAType,
		Interconnect:            info.Interconnect,
		InterconnectTier:        info.InterconnectTier,
		NVLinkLinks:             info.NVLinkLinks,
	}
}

func (info *DiscoveredAcceleratorInfo) toGPUInfo() *GPUInfo {
	if info == nil {
		return nil
	}

	return &GPUInfo{
		NodeName:         info.NodeName,
		GPUsPerNode:      info.AcceleratorsPerNode,
		NodesWithGPUs:    info.NodesWithAccelerators,
		Model:            info.Model,
		VRAMPerGPU:       info.MemoryPerAcceleratorMiB,
		System:           info.SKU,
		MIGEnabled:       info.MIGEnabled,
		MIGProfiles:      cloneMIGProfiles(info.MIGProfiles),
		CloudProvider:    info.CloudProvider,
		RDMAEnabled:      info.RDMAEnabled,
		RDMAType:         info.RDMAType,
		Interconnect:     info.Interconnect,
		InterconnectTier: info.InterconnectTier,
		NVLinkLinks:      info.NVLinkLinks,
	}
}

func cloneMIGProfiles(src map[string]int) map[string]int {
	if len(src) == 0 {
		return nil
	}

	cloned := make(map[string]int, len(src))
	for profile, count := range src {
		cloned[profile] = count
	}
	return cloned
}
