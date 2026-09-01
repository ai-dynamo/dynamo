/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"

// CheckpointInfo is the resolved standalone Snapshot state consumed by
// workload rendering.
type CheckpointInfo struct {
	Enabled          bool
	Exists           bool
	AutomaticCapture bool
	GPUMemoryService *nvidiacomv1alpha1.GPUMemoryServiceSpec
	CheckpointName   string
	Ready            bool
	StartupPolicy    nvidiacomv1alpha1.CheckpointStartupPolicy
	// Empty means the restore pod targets the default main container.
	RestoreTargetContainers []string
	// NativeSnapshot is non-nil once CheckpointName resolves to a standalone
	// Snapshot PodSnapshot.
	NativeSnapshot *ResolvedPodSnapshot
}
