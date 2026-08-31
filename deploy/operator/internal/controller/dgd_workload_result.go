/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"fmt"
	"sort"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
)

type Reason string
type Message string

type Resource interface {
	IsReady() (ready bool, reason string)
	GetName() string
	GetComponentStatuses() map[string]nvidiacomv1beta1.ComponentReplicaStatus
}

// ReconcileResult is the provider-neutral workload outcome consumed by the
// program that owns final DGD status projection.
type ReconcileResult struct {
	State           nvidiacomv1beta1.DGDState
	Reason          Reason
	Message         Message
	ComponentStatus map[string]nvidiacomv1beta1.ComponentReplicaStatus
}

func checkResourcesReadiness(resources []Resource) ReconcileResult {
	// Sort resources by name to ensure deterministic ordering.
	sort.Slice(resources, func(i, j int) bool {
		return resources[i].GetName() < resources[j].GetName()
	})

	var notReadyReasons []string
	componentStatuses := make(map[string]nvidiacomv1beta1.ComponentReplicaStatus)
	for _, resource := range resources {
		ready, reason := resource.IsReady()

		resourceComponentStatuses := resource.GetComponentStatuses()
		for componentName, componentStatus := range resourceComponentStatuses {
			componentStatuses[componentName] = componentStatus
		}

		if !ready {
			notReadyReasons = append(notReadyReasons, fmt.Sprintf("%s: %s", resource.GetName(), reason))
		}
	}

	if len(notReadyReasons) == 0 {
		return ReconcileResult{
			State:           nvidiacomv1beta1.DGDStateSuccessful,
			Reason:          "all_resources_are_ready",
			Message:         Message("All resources are ready"),
			ComponentStatus: componentStatuses,
		}
	}
	return ReconcileResult{
		State:           nvidiacomv1beta1.DGDStatePending,
		Reason:          "some_resources_are_not_ready",
		Message:         Message(fmt.Sprintf("Resources not ready: %s", strings.Join(notReadyReasons, "; "))),
		ComponentStatus: componentStatuses,
	}
}

func applyCheckpointStartupReadiness(
	result ReconcileResult,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) ReconcileResult {
	if result.State == nvidiacomv1beta1.DGDStateFailed {
		return result
	}
	var waiting []string
	for componentName, info := range checkpointInfos {
		if info == nil ||
			!info.Enabled ||
			info.StartupPolicy != nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint ||
			info.Ready {
			continue
		}
		if info.CheckpointName != "" {
			waiting = append(waiting, fmt.Sprintf("%s (%s)", componentName, info.CheckpointName))
		} else {
			waiting = append(waiting, componentName)
		}
	}
	if len(waiting) == 0 {
		return result
	}
	sort.Strings(waiting)
	result.State = nvidiacomv1beta1.DGDStatePending
	result.Reason = reasonWaitingForCheckpoint
	result.Message = Message(fmt.Sprintf("Waiting for checkpoints: %s", strings.Join(waiting, ", ")))
	return result
}
