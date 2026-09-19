/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"strconv"
	"strings"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	corev1 "k8s.io/api/core/v1"
)

// LPUAgentRuntimePartition maps a hybrid Agent's Grove index to its runtime
// ConfigMap row. Both inputs must be non-nil and config must be the immutable
// table stamped on the Pod's template. Inputs are not mutated. Runtime rows,
// rather than scheduler partitions, preserve XT's collapsed PropSync cohorts.
func LPUAgentRuntimePartition(pod *corev1.Pod, config *corev1.ConfigMap) (int, error) {
	// Consume only canonical Grove identity, never deprecated partition/rank annotations.
	rawIndex := pod.Labels[grovecommon.LabelPodCliquePodIndex]
	index, err := strconv.ParseUint(rawIndex, 10, 32)
	if err != nil || strconv.FormatUint(index, 10) != rawIndex {
		return 0, fmt.Errorf("pod %s/%s has invalid Grove Pod index %q", pod.Namespace, pod.Name, rawIndex)
	}
	model := pod.Annotations[lpxv1alpha1.PodModelAnnotation]
	if model == "" {
		return 0, fmt.Errorf("pod %s/%s has no LPU model", pod.Namespace, pod.Name)
	}

	// Read the same three columns used by the hybrid Agent's partition metadata command.
	counts := strings.Split(config.Data["nodes_per_partition"], "\n")
	offsets := strings.Split(config.Data["partition_node_offsets"], "\n")
	models := strings.Split(config.Data["partition_models"], "\n")
	if len(counts) != len(offsets) || len(counts) != len(models) {
		return 0, fmt.Errorf("ConfigMap %s/%s has inconsistent runtime partition columns", config.Namespace, config.Name)
	}

	// Require one unambiguous runtime range for this model and Grove index.
	matched := -1
	for row, rowModel := range models {
		if rowModel != model {
			continue
		}
		count, countErr := strconv.ParseUint(counts[row], 10, 32)
		offset, offsetErr := strconv.ParseUint(offsets[row], 10, 32)
		if countErr != nil || offsetErr != nil || count == 0 {
			return 0, fmt.Errorf("ConfigMap %s/%s has invalid runtime partition row %d", config.Namespace, config.Name, row)
		}
		if index < offset || index-offset >= count {
			continue
		}
		if matched >= 0 {
			return 0, fmt.Errorf("Grove Pod index %d matches multiple runtime partitions for model %q", index, model)
		}
		matched = row
	}
	if matched < 0 {
		return 0, fmt.Errorf("Grove Pod index %d matches no runtime partition for model %q", index, model)
	}
	return matched, nil
}
