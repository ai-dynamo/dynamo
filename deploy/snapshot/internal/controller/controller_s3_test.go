// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/go-logr/logr/testr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

// fakeStore is an in-memory ArtifactStore for controller tests.
type fakeStore struct {
	existsPrefixes map[string]bool
	existsErr      error
	statedPrefix   string
}

func (f *fakeStore) Upload(ctx context.Context, localDir, keyPrefix string) error   { return nil }
func (f *fakeStore) Download(ctx context.Context, keyPrefix, localDir string) error { return nil }
func (f *fakeStore) Remove(ctx context.Context, keyPrefix string) error             { return nil }
func (f *fakeStore) Backend() string                                                { return "fake" }
func (f *fakeStore) Exists(ctx context.Context, keyPrefix string) (bool, error) {
	f.statedPrefix = keyPrefix
	if f.existsErr != nil {
		return false, f.existsErr
	}
	return f.existsPrefixes[keyPrefix], nil
}

func TestReachesIntoPod(t *testing.T) {
	t.Run("pvc agentMount reaches in for neither", func(t *testing.T) {
		w := makeTestController(t) // default: pvc, agentMount, no store
		if w.checkpointReachesIntoPod() {
			t.Error("pvc agentMount checkpoint should not reach into the pod")
		}
		if w.restoreReachesIntoPod() {
			t.Error("pvc agentMount restore should not reach into the pod")
		}
	})

	t.Run("s3 reaches in for restore only", func(t *testing.T) {
		w := makeTestController(t)
		w.store = &fakeStore{} // s3 backend present
		if w.checkpointReachesIntoPod() {
			t.Error("s3 checkpoint stages on the agent and must not reach into the pod")
		}
		if !w.restoreReachesIntoPod() {
			t.Error("s3 restore must reach into the pod to stage the downloaded artifact")
		}
	})

	t.Run("podMount reaches in for both", func(t *testing.T) {
		w := makeTestController(t)
		w.config.Storage.AccessMode = types.StorageAccessModePodMount
		if !w.checkpointReachesIntoPod() || !w.restoreReachesIntoPod() {
			t.Error("podMount must reach into the pod for checkpoint and restore")
		}
	})
}

func TestObjectKeyPrefix(t *testing.T) {
	w := makeTestController(t)
	w.config.Storage.S3.Prefix = "team-a"
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Annotations: map[string]string{snapshotprotocol.CheckpointArtifactVersionAnnotation: "3"},
	}}
	if got := w.objectKeyPrefix(pod, "sha256:abc"); got != "team-a/sha256:abc/versions/3" {
		t.Errorf("objectKeyPrefix = %q", got)
	}

	// Missing version annotation falls back to the default artifact version.
	bare := &corev1.Pod{}
	w.config.Storage.S3.Prefix = ""
	want := "sha256:abc/versions/" + snapshotprotocol.DefaultCheckpointArtifactVersion
	if got := w.objectKeyPrefix(bare, "sha256:abc"); got != want {
		t.Errorf("objectKeyPrefix without version = %q, want %q", got, want)
	}
}

func TestRestoreCheckpointReadyObjectStore(t *testing.T) {
	log := testr.New(t)
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Annotations: map[string]string{snapshotprotocol.CheckpointArtifactVersionAnnotation: "1"},
	}}

	t.Run("ready when manifest present", func(t *testing.T) {
		w := makeTestController(t)
		w.store = &fakeStore{existsPrefixes: map[string]bool{"abc123/versions/1": true}}
		ready, err := w.restoreCheckpointReady(context.Background(), log, pod, "default/p", "abc123", "/unused")
		if err != nil || !ready {
			t.Fatalf("ready=%v err=%v; want true, nil", ready, err)
		}
	})

	t.Run("not ready when artifact absent", func(t *testing.T) {
		w := makeTestController(t)
		w.store = &fakeStore{existsPrefixes: map[string]bool{}}
		ready, err := w.restoreCheckpointReady(context.Background(), log, pod, "default/p", "abc123", "/unused")
		if err != nil || ready {
			t.Fatalf("ready=%v err=%v; want false, nil", ready, err)
		}
	})

	t.Run("propagates store error", func(t *testing.T) {
		w := makeTestController(t)
		w.store = &fakeStore{existsErr: errors.New("boom")}
		if _, err := w.restoreCheckpointReady(context.Background(), log, pod, "default/p", "abc123", "/unused"); err == nil {
			t.Fatal("expected error from store")
		}
	})
}
