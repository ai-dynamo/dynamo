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

// renderCyborgConfigMap renders the hybrid Agent endpoints of either LPU family.
// It lists one Agent per remote runtime partition in compiler order; an all-local
// selection renders an empty list. The workload and plan must be non-nil,
// validated, and describe the selected hybrid model.
// The workload and plan are not mutated. The caller assigns the returned ConfigMap's namespace.
func (w *Workload) renderCyborgConfigMap(plan *MaterializationPlan) (*corev1.ConfigMap, string, error) {
	projection := w.modelProjections[0]
	build := &projection.configuredBuild

	// Cyborg runs a selected prop-sync chain as one runtime partition and
	// addresses it through its first member. XT builds have already collapsed
	// these chains; HX builds keep each member as a physical partition.
	chained := make([]bool, len(build.Partitions))
	for _, position := range projection.propSyncEdges {
		chained[position+1] = true
	}

	// Cyborg supplies the PCS prefix; startup supplies this workload's Grove index.
	serverPrefix := plan.ScalingGroupTemplate + "-${GROVE_PCSG_INDEX}-" + plan.Agents[0].TemplateName + "-"
	servers := make([]string, 0, len(build.Partitions))
	offset := 0
	for index, partition := range build.Partitions {
		if !chained[index] {
			servers = append(servers, serverPrefix+strconv.Itoa(offset))
		}
		offset += partition.effectiveNodeCount()
	}

	return renderRuntimeConfigMap(plan.ResourcePrefix+"-decode", map[string]string{
		"lpu_servers": strings.Join(servers, "\n"),
	})
}
