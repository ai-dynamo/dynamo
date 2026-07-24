//go:build linux

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	crfake "sigs.k8s.io/controller-runtime/pkg/client/fake"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

// snapshotScheme returns a scheme with the operator CRD and core types registered.
func snapshotScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(s))
	require.NoError(t, corev1.AddToScheme(s))
	return s
}

// TestPodSnapshotName verifies the DNS-1123-safe name derivation from Job names.
func TestPodSnapshotName(t *testing.T) {
	tests := []struct {
		name    string
		jobName string
		wantLen int
		wantEq  string // exact match when non-empty
	}{
		{
			name:    "short name passes through unchanged",
			jobName: "my-worker-checkpoint",
			wantEq:  "my-worker-checkpoint",
		},
		{
			name:    "exactly 63 chars passes through unchanged",
			jobName: strings.Repeat("a", 63),
			wantEq:  strings.Repeat("a", 63),
		},
		{
			name:    "64-char name is capped to 63",
			jobName: strings.Repeat("b", 64),
			wantEq:  strings.Repeat("b", 63),
		},
		{
			name:    "100-char name is capped to 63",
			jobName: strings.Repeat("c", 100),
			wantEq:  strings.Repeat("c", 63),
		},
		{
			name:    "trailing hyphen is stripped after truncation",
			jobName: strings.Repeat("e", 62) + "-extra",
			wantLen: 62, // the hyphen at position 62 is stripped
		},
		{
			name:    "capped name is deterministic",
			jobName: strings.Repeat("d", 80),
			wantEq:  strings.Repeat("d", 63),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := podSnapshotName(tt.jobName)
			if tt.wantEq != "" {
				assert.Equal(t, tt.wantEq, got)
			}
			if tt.wantLen > 0 {
				assert.Equal(t, tt.wantLen, len(got), "capped name must be exactly 63 chars")
			}
			// Verify determinism: calling twice yields the same result.
			assert.Equal(t, got, podSnapshotName(tt.jobName), "podSnapshotName must be deterministic")
		})
	}
}

// TestPodSnapshotNameUIDPinned verifies that createPodSnapshot stamps the source pod's UID
// onto spec.source.podRef.uid so the operator rejects a same-named replacement pod.
func TestPodSnapshotNameUIDPinned(t *testing.T) {
	s := snapshotScheme(t)
	podUID := types.UID("pod-uid-abc")

	tests := []struct {
		name    string
		snapName string
	}{
		{name: "short name", snapName: "my-snap"},
		// Name derived from a >63-char job name is truncated; UID must still be pinned.
		{name: "long name from >63-char job", snapName: podSnapshotName(strings.Repeat("x", 80))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			crClient := crfake.NewClientBuilder().WithScheme(s).Build()
			snap, err := createPodSnapshot(context.Background(), crClient, "default", tt.snapName, "my-pod", podUID, "ckpt-123")
			require.NoError(t, err)
			assert.Equal(t, podUID, snap.Spec.Source.PodRef.UID, "source pod UID must be pinned")
			assert.Equal(t, "my-pod", snap.Spec.Source.PodRef.Name)
		})
	}
}

// TestPodSnapshotAlreadyExists verifies that createPodSnapshot returns a clear error when
// a PodSnapshot with the derived name already exists.
func TestPodSnapshotAlreadyExists(t *testing.T) {
	s := snapshotScheme(t)
	existing := &nvidiacomv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{Name: "my-snap", Namespace: "default"},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{
				PodRef: nvidiacomv1alpha1.PodReference{Name: "other-pod"},
			},
		},
	}
	crClient := crfake.NewClientBuilder().WithScheme(s).WithObjects(existing).Build()

	_, err := createPodSnapshot(context.Background(), crClient, "default", "my-snap", "my-pod", "uid-1", "ckpt-1")
	require.Error(t, err)
	assert.True(t, apierrors.IsAlreadyExists(err) || strings.Contains(err.Error(), "already exists"),
		"error should report AlreadyExists: %v", err)
}

// TestWaitForPodSnapshotReady verifies that waitForPodSnapshot returns the PodSnapshot when
// its Ready condition is True.
func TestWaitForPodSnapshotReady(t *testing.T) {
	s := snapshotScheme(t)
	content := "my-content"
	snap := &nvidiacomv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{Name: "snap-1", Namespace: "default"},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{
				PodRef: nvidiacomv1alpha1.PodReference{Name: "pod-1"},
			},
		},
		Status: nvidiacomv1alpha1.PodSnapshotStatus{
			BoundPodSnapshotContentName: &content,
			Conditions: []metav1.Condition{
				{
					Type:               "Ready",
					Status:             metav1.ConditionTrue,
					Reason:             "Captured",
					Message:            "checkpoint captured",
					LastTransitionTime: metav1.Now(),
				},
			},
		},
	}
	crClient := crfake.NewClientBuilder().
		WithScheme(s).
		WithObjects(snap).
		Build()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	result, err := waitForPodSnapshot(ctx, crClient, "default", "snap-1")
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.True(t, nvidiacomv1alpha1.IsPodSnapshotSucceeded(result))
	assert.NotNil(t, result.Status.BoundPodSnapshotContentName)
}

// TestWaitForPodSnapshotFailed verifies that waitForPodSnapshot returns an error surfacing the
// Failed condition's Reason and Message when the PodSnapshot fails.
func TestWaitForPodSnapshotFailed(t *testing.T) {
	s := snapshotScheme(t)
	snap := &nvidiacomv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{Name: "snap-fail", Namespace: "default"},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{
				PodRef: nvidiacomv1alpha1.PodReference{Name: "pod-1"},
			},
		},
		Status: nvidiacomv1alpha1.PodSnapshotStatus{
			Conditions: []metav1.Condition{
				{
					Type:               "Failed",
					Status:             metav1.ConditionTrue,
					Reason:             "AgentError",
					Message:            "CRIU dump failed: no such process",
					LastTransitionTime: metav1.Now(),
				},
			},
		},
	}
	crClient := crfake.NewClientBuilder().
		WithScheme(s).
		WithObjects(snap).
		Build()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := waitForPodSnapshot(ctx, crClient, "default", "snap-fail")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "AgentError", "error must surface the Failed condition Reason")
	assert.Contains(t, err.Error(), "CRIU dump failed: no such process", "error must surface the Failed condition Message")
}

// TestWaitForSourcePodTimeout verifies that waitForSourcePod returns an actionable error when
// the overall context expires before the pod appears.
func TestWaitForSourcePodTimeout(t *testing.T) {
	// Use a fake clientset with no pods — the pod will never appear.
	clientset := k8sfake.NewClientset()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err := waitForSourcePod(ctx, clientset, "default", "my-job", "job-uid-1")
	require.Error(t, err)
	// Must mention the job so the caller knows which job is stuck.
	assert.Contains(t, err.Error(), "my-job", "timeout error must name the job")
}

// TestWaitForSourcePodUnscheduled verifies that waitForSourcePod keeps waiting when the pod
// exists but has not yet been scheduled (Spec.NodeName == ""), then times out with an actionable
// error rather than returning the unscheduled pod.
func TestWaitForSourcePodUnscheduled(t *testing.T) {
	controllerTrue := true
	jobUID := types.UID("job-uid-2")
	unscheduledPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-job-pod",
			Namespace: "default",
			Labels:    map[string]string{"batch.kubernetes.io/job-name": "my-job"},
			OwnerReferences: []metav1.OwnerReference{
				{UID: jobUID, Controller: &controllerTrue},
			},
		},
		Spec: corev1.PodSpec{
			// NodeName is empty — not yet scheduled.
			Containers: []corev1.Container{{Name: "main", Image: "test"}},
		},
	}
	clientset := k8sfake.NewClientset(unscheduledPod)
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	_, err := waitForSourcePod(ctx, clientset, "default", "my-job", jobUID)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "my-job", "timeout error must name the job")
}

// TestWaitForPodSnapshotContextDeadline verifies that waitForPodSnapshot propagates a context
// deadline exceeded error when the PodSnapshot never reaches a terminal state.
func TestWaitForPodSnapshotContextDeadline(t *testing.T) {
	s := snapshotScheme(t)
	// Non-terminal PodSnapshot: no Ready or Failed condition.
	snap := &nvidiacomv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{Name: "snap-pending", Namespace: "default"},
		Spec: nvidiacomv1alpha1.PodSnapshotSpec{
			Source: nvidiacomv1alpha1.PodSnapshotSource{
				PodRef: nvidiacomv1alpha1.PodReference{Name: "pod-1"},
			},
		},
	}
	crClient := crfake.NewClientBuilder().WithScheme(s).WithObjects(snap).Build()

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	_, err := waitForPodSnapshot(ctx, crClient, "default", "snap-pending")
	require.Error(t, err)
}

