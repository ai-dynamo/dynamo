// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import "testing"

func TestApplyRestoreTargetMetadataClearsPerContainerState(t *testing.T) {
	labels := map[string]string{
		CheckpointSourceLabel: "true",
		CheckpointIDLabel:     "old",
	}
	annotations := map[string]string{
		CheckpointArtifactVersionAnnotation:                  "old",
		CheckpointStatusAnnotationPrefix + "main":            "completed",
		CheckpointStatusAnnotationPrefix + "engine-0":        "failed",
		RestoreStatusAnnotationPrefix + "main":               "failed",
		RestoreStatusAnnotationPrefix + "engine-1":           "in_progress",
		RestoreContainerIDAnnotationPrefix + "main":          "dead-container",
		RestoreContainerIDAnnotationPrefix + "engine-0":      "older-container",
		"unrelated/annotation":                               "keep-me",
	}

	ApplyRestoreTargetMetadata(labels, annotations, true, "hash", "2")

	if labels[RestoreTargetLabel] != "true" {
		t.Fatalf("expected restore target label, got %#v", labels)
	}
	if labels[CheckpointIDLabel] != "hash" {
		t.Fatalf("expected checkpoint hash label, got %#v", labels)
	}
	if _, ok := labels[CheckpointSourceLabel]; ok {
		t.Fatalf("checkpoint source label was not cleared: %#v", labels)
	}
	if annotations[CheckpointArtifactVersionAnnotation] != "2" {
		t.Fatalf("expected checkpoint artifact version annotation, got %#v", annotations)
	}
	for key := range annotations {
		switch key {
		case CheckpointArtifactVersionAnnotation, "unrelated/annotation":
			continue
		}
		t.Fatalf("expected per-container annotation %q to be cleared: %#v", key, annotations)
	}
	if annotations["unrelated/annotation"] != "keep-me" {
		t.Fatalf("expected unrelated annotation to be preserved: %#v", annotations)
	}
}

func TestApplyRestoreTargetMetadataDisabledClearsState(t *testing.T) {
	labels := map[string]string{
		RestoreTargetLabel: "true",
		CheckpointIDLabel:  "hash",
	}
	annotations := map[string]string{
		CheckpointArtifactVersionAnnotation:             "2",
		CheckpointStatusAnnotationPrefix + "main":       "completed",
		RestoreStatusAnnotationPrefix + "main":          "failed",
		RestoreContainerIDAnnotationPrefix + "main":     "dead-container",
	}

	ApplyRestoreTargetMetadata(labels, annotations, false, "", "")

	if _, ok := labels[RestoreTargetLabel]; ok {
		t.Fatalf("restore target label was not cleared: %#v", labels)
	}
	if _, ok := labels[CheckpointIDLabel]; ok {
		t.Fatalf("checkpoint hash label was not cleared: %#v", labels)
	}
	if _, ok := annotations[CheckpointArtifactVersionAnnotation]; ok {
		t.Fatalf("checkpoint artifact version annotation was not cleared: %#v", annotations)
	}
	for key := range annotations {
		t.Fatalf("expected all per-container annotations cleared, still has %q: %#v", key, annotations)
	}
}

func TestParseCheckpointContainers(t *testing.T) {
	t.Run("ordered list", func(t *testing.T) {
		names, err := ParseCheckpointContainers(map[string]string{
			CheckpointContainersAnnotation: "engine-0, engine-1 ,main",
		})
		if err != nil {
			t.Fatalf("expected parse ok, got %v", err)
		}
		if len(names) != 3 || names[0] != "engine-0" || names[1] != "engine-1" || names[2] != "main" {
			t.Fatalf("unexpected names: %#v", names)
		}
	})

	t.Run("rejects missing", func(t *testing.T) {
		if _, err := ParseCheckpointContainers(nil); err == nil {
			t.Fatalf("expected error on missing annotation")
		}
	})

	t.Run("rejects empty entry", func(t *testing.T) {
		if _, err := ParseCheckpointContainers(map[string]string{
			CheckpointContainersAnnotation: "main,,engine",
		}); err == nil {
			t.Fatalf("expected error on empty entry")
		}
	})

	t.Run("rejects duplicate", func(t *testing.T) {
		if _, err := ParseCheckpointContainers(map[string]string{
			CheckpointContainersAnnotation: "main,main",
		}); err == nil {
			t.Fatalf("expected error on duplicate entry")
		}
	})
}

func TestContainerCheckpointPath(t *testing.T) {
	if got := ContainerCheckpointPath("/checkpoints/abc/versions/2", "engine-0"); got != "/checkpoints/abc/versions/2/containers/engine-0" {
		t.Fatalf("unexpected container path: %s", got)
	}
	if got := ContainerCheckpointPath("/checkpoints/abc/versions/2/", "main"); got != "/checkpoints/abc/versions/2/containers/main" {
		t.Fatalf("unexpected container path with trailing slash: %s", got)
	}
}
