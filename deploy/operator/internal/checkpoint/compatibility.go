/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"errors"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	checkpointInterPodCompatibilityMessage = "Snapshot with gpuMemoryService.mode=InterPod is unsupported"
	checkpointFailoverCompatibilityMessage = "Snapshot with active/passive failover is temporarily unsupported"
)

// ValidateCheckpointCompatibility returns unsupported checkpoint combinations
// in stable policy order.
func ValidateCheckpointCompatibility(experimental *nvidiacomv1beta1.ExperimentalSpec) []error {
	if experimental == nil ||
		experimental.Checkpoint == nil || !experimental.Checkpoint.Enabled {
		return nil
	}

	var violations []error
	if experimental.GPUMemoryService != nil &&
		experimental.GPUMemoryService.Mode == nvidiacomv1beta1.GMSModeInterPod {
		violations = append(violations, errors.New(checkpointInterPodCompatibilityMessage))
	}
	if experimental.Failover != nil {
		violations = append(violations, errors.New(checkpointFailoverCompatibilityMessage))
	}

	return violations
}
