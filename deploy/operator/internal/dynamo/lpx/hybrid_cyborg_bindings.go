/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

const (
	// agentRDMAPort is the port on which every hybrid Agent serves RDMA and Cyborg dials it.
	agentRDMAPort = 19877
	// rdmaPortEnvVar publishes agentRDMAPort to hybrid Agent and Cyborg containers.
	rdmaPortEnvVar = "LPX_RDMA_PORT"
	// lpuRDMAAddrsEnvVar lists Cyborg's remote Agent RDMA endpoints as comma-separated host:port pairs.
	lpuRDMAAddrsEnvVar = "LPX_LPU_RDMA_ADDRS"
)

// cyborgAgentOffsets lists the first Agent Pod index of each remote runtime
// partition that Cyborg addresses, in compiler order. An all-local selection
// returns an empty list. The workload must describe a validated hybrid model.
func (w *Workload) cyborgAgentOffsets() []int {
	projection := w.modelProjections[0]
	build := &projection.configuredBuild

	// Cyborg runs a selected prop-sync chain as one runtime partition and
	// addresses it through its first member. XT builds have already collapsed
	// these chains; HX builds keep each member as a physical partition.
	chained := make([]bool, len(build.partitions))
	for _, position := range projection.propSyncEdges {
		chained[position+1] = true
	}

	offsets := make([]int, 0, len(build.partitions))
	offset := 0
	for index, partition := range build.partitions {
		if !chained[index] {
			offsets = append(offsets, offset)
		}
		offset += partition.effectiveNodeCount()
	}
	return offsets
}

// renderCyborgConfigMap renders the hybrid Agent endpoints of either LPU family.
// It lists one Agent per remote runtime partition in compiler order; an all-local
// selection renders an empty list. The workload and plan must be non-nil,
// validated, and describe the selected hybrid model.
// The workload and plan are not mutated. The caller assigns the returned ConfigMap's namespace.
//
// lpu_servers is a temporary bridge for Cyborg images whose
// `cyborg-entrypoint --expand-hosts` expands these names into Agent addresses.
// Remove it once Cyborg reads LPX_LPU_RDMA_ADDRS.
func (w *Workload) renderCyborgConfigMap(plan *MaterializationPlan) (*corev1.ConfigMap, string, error) {
	// Cyborg supplies the PCS prefix; startup supplies this workload's Grove index.
	serverPrefix := plan.ScalingGroupTemplate + "-${GROVE_PCSG_INDEX}-" + plan.Agents[0].TemplateName + "-"
	offsets := w.cyborgAgentOffsets()
	servers := make([]string, len(offsets))
	for index, offset := range offsets {
		servers[index] = serverPrefix + strconv.Itoa(offset)
	}

	return renderRuntimeConfigMap(plan.ResourcePrefix+"-decode", map[string]string{
		"lpu_servers": strings.Join(servers, "\n"),
	})
}

// cyborgAgentRDMAAddrs renders Cyborg's remote Agent RDMA endpoints in the
// order of renderCyborgConfigMap. Each host uses Kubernetes dependent-variable
// expansion of the Grove variables, which Grove places before template env;
// GROVE_PCSG_NAME is <pcs>-<pcsIndex>-<scalingGroupTemplate>. The workload and
// plan must be non-nil, validated, and describe the selected hybrid model.
func (w *Workload) cyborgAgentRDMAAddrs(plan *MaterializationPlan) string {
	hostPrefix := "$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-" + plan.Agents[0].TemplateName + "-"
	hostSuffix := ".$(GROVE_HEADLESS_SERVICE):" + strconv.Itoa(agentRDMAPort)
	offsets := w.cyborgAgentOffsets()
	addrs := make([]string, len(offsets))
	for index, offset := range offsets {
		addrs[index] = hostPrefix + strconv.Itoa(offset) + hostSuffix
	}
	return strings.Join(addrs, ",")
}
