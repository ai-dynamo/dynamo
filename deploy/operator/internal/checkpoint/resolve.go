/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"k8s.io/apimachinery/pkg/types"
)

// SnapshotJobReference identifies the immutable automatic-capture job whose
// eventual PodSnapshot may restore future workload Pods.
type SnapshotJobReference struct {
	Name string
	UID  types.UID
}

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
	// AutomaticSnapshotJob remains stable before and after capture completion,
	// preventing Snapshot readiness from changing an Immediate workload template.
	AutomaticSnapshotJob *SnapshotJobReference
}
