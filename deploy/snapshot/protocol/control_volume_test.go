// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestEnsureControlVolume(t *testing.T) {
	t.Run("adds volume mount and env from empty", func(t *testing.T) {
		ps := &corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}}
		EnsureControlVolume(ps, &ps.Containers[0])

		if len(ps.Volumes) != 1 || ps.Volumes[0].Name != SnapshotControlVolumeName || ps.Volumes[0].EmptyDir == nil {
			t.Fatalf("expected one %s emptyDir volume, got %#v", SnapshotControlVolumeName, ps.Volumes)
		}
		c := ps.Containers[0]
		if len(c.VolumeMounts) != 1 || c.VolumeMounts[0].Name != SnapshotControlVolumeName || c.VolumeMounts[0].MountPath != SnapshotControlMountPath {
			t.Fatalf("expected one %s mount at %s, got %#v", SnapshotControlVolumeName, SnapshotControlMountPath, c.VolumeMounts)
		}
		if c.VolumeMounts[0].SubPath != "main" {
			t.Fatalf("expected subPath=%q, got %q", "main", c.VolumeMounts[0].SubPath)
		}
		assertEnv(t, c.Env, SnapshotControlDirEnv, SnapshotControlMountPath)
		assertEnv(t, c.Env, TorchC10dRendezvousFileEnv, SnapshotControlMountPath+"/"+TorchC10dRendezvousFile)
		assertEnv(t, c.Env, NCCLCheckpointKVSPathEnv, SnapshotControlMountPath+"/"+NCCLCheckpointKVSFile)
	})

	t.Run("per-container subPath isolates multi-container pods", func(t *testing.T) {
		ps := &corev1.PodSpec{Containers: []corev1.Container{
			{Name: "engine-0"},
			{Name: "engine-1"},
		}}
		EnsureControlVolume(ps, &ps.Containers[0])
		EnsureControlVolume(ps, &ps.Containers[1])

		if len(ps.Volumes) != 1 {
			t.Fatalf("expected single shared emptyDir, got %#v", ps.Volumes)
		}
		if got := ps.Containers[0].VolumeMounts[0].SubPath; got != "engine-0" {
			t.Fatalf("engine-0 subPath=%q", got)
		}
		if got := ps.Containers[1].VolumeMounts[0].SubPath; got != "engine-1" {
			t.Fatalf("engine-1 subPath=%q", got)
		}
	})

	t.Run("idempotent", func(t *testing.T) {
		ps := &corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}}
		EnsureControlVolume(ps, &ps.Containers[0])
		EnsureControlVolume(ps, &ps.Containers[0])
		c := ps.Containers[0]
		if len(ps.Volumes) != 1 || len(c.VolumeMounts) != 1 || len(c.Env) != 3 {
			t.Fatalf("expected single volume/mount/env after two calls, got volumes=%d mounts=%d env=%d", len(ps.Volumes), len(c.VolumeMounts), len(c.Env))
		}
	})

	t.Run("nil pod spec no-op", func(t *testing.T) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("expected no panic, got %v", r)
			}
		}()
		EnsureControlVolume(nil, &corev1.Container{})
	})

	t.Run("nil container no-op", func(t *testing.T) {
		ps := &corev1.PodSpec{}
		EnsureControlVolume(ps, nil)
		if len(ps.Volumes) != 0 {
			t.Fatalf("expected no volumes when container is nil, got %#v", ps.Volumes)
		}
	})

	t.Run("preserves existing entries", func(t *testing.T) {
		ps := &corev1.PodSpec{
			Volumes: []corev1.Volume{{
				Name:         "other",
				VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
			}},
			Containers: []corev1.Container{{
				Name:         "main",
				VolumeMounts: []corev1.VolumeMount{{Name: "other", MountPath: "/other"}},
				Env:          []corev1.EnvVar{{Name: "OTHER", Value: "x"}},
			}},
		}
		EnsureControlVolume(ps, &ps.Containers[0])
		c := ps.Containers[0]
		if len(ps.Volumes) != 2 || len(c.VolumeMounts) != 2 || len(c.Env) != 4 {
			t.Fatalf("expected existing + control entries, got volumes=%#v mounts=%#v env=%#v", ps.Volumes, c.VolumeMounts, c.Env)
		}
	})
}

func assertEnv(t *testing.T, env []corev1.EnvVar, name, value string) {
	t.Helper()
	for _, e := range env {
		if e.Name == name {
			if e.Value != value {
				t.Fatalf("expected env %s=%s, got %s", name, value, e.Value)
			}
			return
		}
	}
	t.Fatalf("missing env %s in %#v", name, env)
}
