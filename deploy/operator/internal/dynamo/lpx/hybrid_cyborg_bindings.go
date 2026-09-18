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

// RenderCyborgConfigMap renders the XT hybrid Agent endpoints.
// The workload and plan must be non-nil, validated, and describe the selected hybrid model.
// The workload and plan are not mutated.
func (w *SelectedWorkload) RenderCyborgConfigMap(
	namespace string,
	plan *MaterializationPlan,
) (*corev1.ConfigMap, error) {
	build := &w.modelProjections[0].configuredBuild

	// Cyborg supplies the PCS prefix; startup supplies this engine's Grove index.
	serverPrefix := lpxScalingGroupTemplateName + "-${GROVE_PCSG_INDEX}-" + plan.Agents[0].TemplateName + "-"
	servers := make([]string, len(build.Partitions))
	offset := 0
	for index, partition := range build.Partitions {
		servers[index] = serverPrefix + strconv.Itoa(offset)
		offset += partition.effectiveNodeCount()
	}

	return renderRuntimeConfigMap(namespace, plan.PodCliqueSetName+"-decode", map[string]string{
		"lpu_servers": strings.Join(servers, "\n"),
	})
}
