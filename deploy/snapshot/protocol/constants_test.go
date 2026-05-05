// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"reflect"
	"strings"
	"testing"
)

func TestApplyRestoreTargetMetadata(t *testing.T) {
	labels := map[string]string{
		CheckpointSourceLabel: "true",
		CheckpointIDLabel:     "old",
	}
	annotations := map[string]string{
		CheckpointArtifactVersionAnnotation:             "old",
		CheckpointStatusAnnotation:                      "completed",
		RestoreStatusAnnotationPrefix + "main":          "failed",
		RestoreStatusAnnotationPrefix + "engine-1":      "completed",
		RestoreContainerIDAnnotationPrefix + "main":     "dead-container",
		RestoreContainerIDAnnotationPrefix + "engine-1": "dead-container",
		"nvidia.com/snapshot-restore-status":            "completed",
		"nvidia.com/snapshot-restore-container-id":      "dead-container",
		// Preserve the target-containers annotation across ApplyRestoreTargetMetadata.
		TargetContainersAnnotation: "main",
	}

	ApplyRestoreTargetMetadata(labels, annotations, true, "hash", "2")

	if labels[CheckpointIDLabel] != "hash" {
		t.Fatalf("expected checkpoint hash label, got %#v", labels)
	}
	if _, ok := labels[CheckpointSourceLabel]; ok {
		t.Fatalf("checkpoint source label was not cleared: %#v", labels)
	}
	if annotations[CheckpointArtifactVersionAnnotation] != "2" {
		t.Fatalf("expected checkpoint artifact version annotation, got %#v", annotations)
	}
	if _, ok := annotations[CheckpointStatusAnnotation]; ok {
		t.Fatalf("checkpoint status annotation was not cleared: %#v", annotations)
	}
	for _, key := range []string{
		RestoreStatusAnnotationPrefix + "main",
		RestoreStatusAnnotationPrefix + "engine-1",
		RestoreContainerIDAnnotationPrefix + "main",
		RestoreContainerIDAnnotationPrefix + "engine-1",
		"nvidia.com/snapshot-restore-status",
		"nvidia.com/snapshot-restore-container-id",
	} {
		if _, ok := annotations[key]; ok {
			t.Fatalf("restore annotation %s was not cleared: %#v", key, annotations)
		}
	}
	if got := annotations[TargetContainersAnnotation]; got != "main" {
		t.Fatalf("target-containers annotation must be preserved, got %q", got)
	}
}

func TestApplyRestoreTargetMetadataDisabledClearsState(t *testing.T) {
	labels := map[string]string{
		CheckpointIDLabel: "hash",
	}
	annotations := map[string]string{
		CheckpointArtifactVersionAnnotation:         "2",
		CheckpointStatusAnnotation:                  "completed",
		RestoreStatusAnnotationPrefix + "main":      "failed",
		RestoreContainerIDAnnotationPrefix + "main": "dead-container",
	}

	ApplyRestoreTargetMetadata(labels, annotations, false, "", "")

	if _, ok := labels[CheckpointIDLabel]; ok {
		t.Fatalf("checkpoint hash label was not cleared: %#v", labels)
	}
	if _, ok := annotations[CheckpointArtifactVersionAnnotation]; ok {
		t.Fatalf("checkpoint artifact version annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[CheckpointStatusAnnotation]; ok {
		t.Fatalf("checkpoint status annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreStatusAnnotationPrefix+"main"]; ok {
		t.Fatalf("per-container restore status was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreContainerIDAnnotationPrefix+"main"]; ok {
		t.Fatalf("per-container restore container id was not cleared: %#v", annotations)
	}
}

func TestParseTargetContainers(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		want    []string
		wantErr bool
	}{
		{name: "empty", in: "", want: nil},
		{name: "whitespace", in: "   ", want: nil},
		{name: "single", in: "main", want: []string{"main"}},
		{name: "two", in: "engine-0,engine-1", want: []string{"engine-0", "engine-1"}},
		{name: "whitespace preserved in split", in: " engine-0 , engine-1 ", want: []string{"engine-0", "engine-1"}},
		{name: "duplicate rejected", in: "a,a", wantErr: true},
		{name: "empty token rejected", in: "a,,b", wantErr: true},
		{name: "trailing comma rejected", in: "a,", wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseTargetContainers(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %#v, want %#v", got, tc.want)
			}
		})
	}
}

func TestFormatTargetContainers(t *testing.T) {
	if got := FormatTargetContainers([]string{"a", " b ", "", "c"}); got != "a,b,c" {
		t.Fatalf("got %q", got)
	}
	if got := FormatTargetContainers(nil); got != "" {
		t.Fatalf("got %q", got)
	}
}

func TestTargetContainersFromAnnotationsMissing(t *testing.T) {
	_, err := TargetContainersFromAnnotations(map[string]string{}, 1, 1)
	if err == nil {
		t.Fatalf("expected missing annotation error")
	}
	if _, err := TargetContainersFromAnnotations(map[string]string{TargetContainersAnnotation: ""}, 1, 1); err == nil {
		t.Fatalf("expected missing annotation error for empty value")
	}
}

func TestTargetContainersFromAnnotationsBounds(t *testing.T) {
	annotations := map[string]string{TargetContainersAnnotation: "engine-0,engine-1"}
	if _, err := TargetContainersFromAnnotations(annotations, 1, 1); err == nil {
		t.Fatalf("expected max-1 enforcement to reject 2 containers")
	}
	got, err := TargetContainersFromAnnotations(annotations, 1, 0)
	if err != nil {
		t.Fatalf("unexpected error for unbounded max: %v", err)
	}
	if !reflect.DeepEqual(got, []string{"engine-0", "engine-1"}) {
		t.Fatalf("got %#v", got)
	}
	if _, err := TargetContainersFromAnnotations(map[string]string{TargetContainersAnnotation: "a,a"}, 1, 0); err == nil {
		t.Fatalf("expected dup rejection")
	}
}

func TestRestoreStatusAnnotations(t *testing.T) {
	got, err := RestoreStatusAnnotations("engine-1", RestoreStatusCompleted, "container-id")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := map[string]string{
		RestoreStatusAnnotationPrefix + "engine-1":      RestoreStatusCompleted,
		RestoreContainerIDAnnotationPrefix + "engine-1": "container-id",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %#v, want %#v", got, want)
	}
}

func TestRestoreStatusAnnotationsRejectsInvalidContainerName(t *testing.T) {
	_, err := RestoreStatusAnnotations(strings.Repeat("a", 200), RestoreStatusInProgress, "container-id")
	if err == nil {
		t.Fatalf("expected invalid annotation key error")
	}
	if !strings.Contains(err.Error(), "restore status annotation key") {
		t.Fatalf("unexpected error: %v", err)
	}
}
