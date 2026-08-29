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

// Intra-pod failover clones the captured "main" container into engine-0..N, so
// the agent's default same-name restore cannot resolve a destination and fails
// preflight with: restore pod has no destination container named "main".
func TestApplyRestoreContainerMap(t *testing.T) {
	t.Run("failover clones get an explicit map", func(t *testing.T) {
		annotations := map[string]string{}
		if err := ApplyRestoreContainerMap(annotations, "main", []string{"engine-0", "engine-1"}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := "main=engine-0,main=engine-1"
		if got := annotations[RestoreContainerMapAnnotation]; got != want {
			t.Fatalf("%s = %q, want %q", RestoreContainerMapAnnotation, got, want)
		}
	})

	t.Run("single same-named destination stays on the agent default", func(t *testing.T) {
		annotations := map[string]string{RestoreContainerMapAnnotation: "stale=value"}
		if err := ApplyRestoreContainerMap(annotations, "main", []string{"main"}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, ok := annotations[RestoreContainerMapAnnotation]; ok {
			t.Fatalf("annotation should be cleared for the same-name case: %v", annotations)
		}
	})

	t.Run("unknown capture source clears the map", func(t *testing.T) {
		annotations := map[string]string{RestoreContainerMapAnnotation: "stale=value"}
		if err := ApplyRestoreContainerMap(annotations, "", []string{"engine-0"}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, ok := annotations[RestoreContainerMapAnnotation]; ok {
			t.Fatalf("annotation should be cleared without a capture source: %v", annotations)
		}
	})
}
