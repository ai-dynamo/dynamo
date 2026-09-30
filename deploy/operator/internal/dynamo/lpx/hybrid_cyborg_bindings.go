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

// renderCyborgConfigMap renders the XT hybrid Agent endpoints.
// The workload must be non-nil, named, and describe the selected hybrid model.
// The workload is not mutated. The caller assigns the returned ConfigMap's namespace.
func (w *Workload) renderCyborgConfigMap() (*corev1.ConfigMap, string, error) {
	build := &w.models[0].component.configuredBuild

	// Cyborg supplies the PCS prefix; startup supplies this workload's Grove index.
	serverPrefix := w.scalingGroupTemplate + "-${GROVE_PCSG_INDEX}-" + w.models[0].agentTemplate + "-"
	servers := make([]string, len(build.partitions))
	offset := 0
	for index, partition := range build.partitions {
		servers[index] = serverPrefix + strconv.Itoa(offset)
		offset += partition.effectiveNodeCount()
	}

	return renderRuntimeConfigMap(w.resourcePrefix+"-decode", map[string]string{
		"lpu_servers": strings.Join(servers, "\n"),
	})
}
