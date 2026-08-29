// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpointjob

import "testing"

// The snapshot agent gates every restore on nvidia.com/restore-from
// (NodeController.restorePodRequested). Without it a correctly labelled restore
// target is ignored and its containers sit in standby forever.
func TestApplyRestoreTargetMetadataSetsRestoreFromAnnotation(t *testing.T) {
	labels := map[string]string{}
	annotations := map[string]string{}

	ApplyRestoreTargetMetadata(labels, annotations, true, "abc123", "1")

	if got := annotations[RestoreFromAnnotation]; got != "checkpoint-abc123" {
		t.Fatalf("%s = %q, want %q", RestoreFromAnnotation, got, "checkpoint-abc123")
	}
	if labels[RestoreTargetLabel] != labelValueTrue {
		t.Fatalf("restore target label not set: %v", labels)
	}
	if labels[CheckpointIDLabel] != "abc123" {
		t.Fatalf("checkpoint id label not set: %v", labels)
	}

	// Disabling must clear it so a recycled pod template is not left pointing
	// at a stale PodSnapshot.
	ApplyRestoreTargetMetadata(labels, annotations, false, "", "")
	if _, ok := annotations[RestoreFromAnnotation]; ok {
		t.Fatalf("%s should be cleared when restore is disabled", RestoreFromAnnotation)
	}
}
