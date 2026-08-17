// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshottypes "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestSourceMountsForContainer(t *testing.T) {
	pod := &corev1.Pod{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name: "main",
					VolumeMounts: []corev1.VolumeMount{
						{Name: "model", MountPath: "/model-cache", ReadOnly: true, SubPath: "weights"},
						{Name: "scratch", MountPath: "/scratch"},
						{Name: "config", MountPath: "/etc/dynamo", SubPathExpr: "$(POD_NAME)"},
						{Name: "unknown", MountPath: "/unknown"},
						{Name: "snapshot-control", MountPath: "/snapshot-control"},
						{Name: "checkpoint-storage", MountPath: "/checkpoints"},
						{Name: "kube-api-access-abc", MountPath: "/var/run/secrets/kubernetes.io/serviceaccount"},
						{Name: "custom-token", MountPath: "/var/run/secrets/custom"},
					},
				},
				{Name: "sidecar", VolumeMounts: []corev1.VolumeMount{{Name: "model", MountPath: "/other"}}},
			},
			Volumes: []corev1.Volume{
				{Name: "model", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "model-pvc"}}},
				{Name: "scratch", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
				{Name: "config", VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: "dynamo-config"}}}},
				{Name: "snapshot-control", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
				{Name: "checkpoint-storage", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "snapshot-pvc"}}},
				{Name: "kube-api-access-abc", VolumeSource: corev1.VolumeSource{Projected: &corev1.ProjectedVolumeSource{
					Sources: []corev1.VolumeProjection{
						{ServiceAccountToken: &corev1.ServiceAccountTokenProjection{Path: "token"}},
						{ConfigMap: &corev1.ConfigMapProjection{
							LocalObjectReference: corev1.LocalObjectReference{Name: "kube-root-ca.crt"},
							Items:                []corev1.KeyToPath{{Key: "ca.crt", Path: "ca.crt"}},
						}},
						{DownwardAPI: &corev1.DownwardAPIProjection{Items: []corev1.DownwardAPIVolumeFile{{
							Path:     "namespace",
							FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"},
						}}}},
					},
				}}},
				{Name: "custom-token", VolumeSource: corev1.VolumeSource{Projected: &corev1.ProjectedVolumeSource{
					Sources: []corev1.VolumeProjection{{ServiceAccountToken: &corev1.ServiceAccountTokenProjection{
						Path:     "token",
						Audience: "custom-audience",
					}}},
				}}},
			},
		},
	}

	mounts := sourceMountsForContainer(pod, "main")

	require.Len(t, mounts, 5)
	assert.Equal(t, nvidiacomv1alpha1.CheckpointSourceMount{
		Path:         "/model-cache",
		Volume:       "model",
		VolumeSource: "PersistentVolumeClaim/model-pvc",
	}, mounts[0])
	assert.Equal(t, "EmptyDir", mounts[1].VolumeSource)
	assert.Equal(t, "Volume/unknown", mounts[3].VolumeSource)
	assert.Equal(t, "Projected", mounts[4].VolumeSource)
}

func TestCheckpointSourceFromManifest(t *testing.T) {
	createdAt := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	manifest := &snapshottypes.CheckpointManifest{
		CheckpointID: "checkpoint-1",
		CreatedAt:    createdAt,
		K8s: snapshottypes.SourcePodManifest{
			SourceNode:   "node-a",
			PodNamespace: "inference",
			PodName:      "worker-0",
			Mounts: []nvidiacomv1alpha1.CheckpointSourceMount{
				{Path: "/model-cache", Volume: "model", VolumeSource: "PersistentVolumeClaim/model-pvc"},
			},
		},
		CUDA: snapshottypes.CUDAManifest{SourceGPUUUIDs: []string{"GPU-a", "", "GPU-b"}},
	}

	source := checkpointSourceFromManifest(manifest)

	require.NotNil(t, source)
	require.NotNil(t, source.Hardware)
	require.NotNil(t, source.Hardware.GPUCount)
	assert.Equal(t, int32(2), *source.Hardware.GPUCount)
	assert.Equal(t, []nvidiacomv1alpha1.CheckpointSourceGPU{{UUID: "GPU-a"}, {UUID: "GPU-b"}}, source.Hardware.GPUs)
	assert.Equal(t, "node-a", source.Node)
	require.Len(t, source.Mounts, 1)
	require.NotNil(t, source.MountCount)
	assert.Equal(t, int32(1), *source.MountCount)
}

func TestCheckpointSourceFromManifestKeepsAvailableFacts(t *testing.T) {
	manifest := &snapshottypes.CheckpointManifest{
		CheckpointID: "checkpoint-1",
		K8s:          snapshottypes.SourcePodManifest{SourceNode: "node-a"},
	}

	source := checkpointSourceFromManifest(manifest)

	require.NotNil(t, source)
	assert.Nil(t, source.Hardware)
	assert.Equal(t, "node-a", source.Node)
	assert.Nil(t, source.MountCount)
}

func TestCheckpointSourceFromManifestKeepsMountOnlyFacts(t *testing.T) {
	manifest := &snapshottypes.CheckpointManifest{
		CheckpointID: "checkpoint-1",
		K8s: snapshottypes.SourcePodManifest{
			Mounts: []nvidiacomv1alpha1.CheckpointSourceMount{{
				Path:         "/model-cache",
				Volume:       "model",
				VolumeSource: "PersistentVolumeClaim/model-pvc",
			}},
		},
	}

	source := checkpointSourceFromManifest(manifest)

	require.NotNil(t, source)
	require.Len(t, source.Mounts, 1)
	require.NotNil(t, source.MountCount)
	assert.Equal(t, int32(1), *source.MountCount)
}

func TestCheckpointSourceFromManifestKeepsKnownZeroMountCount(t *testing.T) {
	manifest := &snapshottypes.CheckpointManifest{
		CheckpointID: "checkpoint-1",
		K8s:          snapshottypes.SourcePodManifest{Mounts: []nvidiacomv1alpha1.CheckpointSourceMount{}},
	}

	source := checkpointSourceFromManifest(manifest)

	require.NotNil(t, source)
	require.NotNil(t, source.MountCount)
	assert.Zero(t, *source.MountCount)
}

func TestCheckpointSourceReadFailurePreservesReady(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	w := makeNodeController(t, &fakeCheckpointer{}, content)

	require.NoError(t, w.setSnapshotContentSucceeded(context.Background(), content, t.TempDir()))

	updated := getContent(t, w, content.Name)
	assert.True(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
	assert.False(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionFailed))
	assert.Nil(t, updated.Status.Source)
}

func TestSnapshotContentSuccessPublishesSourceWithOnePatchAndNoGet(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	getCalls := 0
	statusPatches := 0
	funcs := interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
			getCalls++
			return c.Get(ctx, key, obj, opts...)
		},
		SubResourcePatch: func(ctx context.Context, c client.Client, sub string, obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption) error {
			statusPatches++
			return c.SubResource(sub).Patch(ctx, obj, patch, opts...)
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, content)
	checkpointPath := t.TempDir()
	require.NoError(t, snapshottypes.WriteManifest(checkpointPath, &snapshottypes.CheckpointManifest{
		CheckpointID: "checkpoint-1",
		K8s:          snapshottypes.SourcePodManifest{SourceNode: "node-a"},
	}))

	require.NoError(t, w.setSnapshotContentSucceeded(context.Background(), content, checkpointPath))
	assert.Zero(t, getCalls)
	assert.Equal(t, 1, statusPatches)

	updated := getContent(t, w, content.Name)
	assert.True(t, meta.IsStatusConditionTrue(updated.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
	require.NotNil(t, updated.Status.Source)
	assert.Equal(t, "node-a", updated.Status.Source.Node)
}

func TestReadyCheckpointSourceRetriesWithoutRevalidatingCheckpointID(t *testing.T) {
	content := makeWorkOrder("podsnapshotcontent-x", "node-a", "x")
	meta.SetStatusCondition(&content.Status.Conditions, metav1.Condition{
		Type:   nvidiacomv1alpha1.PodSnapshotConditionReady,
		Status: metav1.ConditionTrue,
		Reason: "Captured",
	})
	pod := makeSourcePod("Bad_ID")
	failuresRemaining := 1
	funcs := interceptor.Funcs{
		SubResourcePatch: func(ctx context.Context, c client.Client, sub string, obj client.Object, patch client.Patch, opts ...client.SubResourcePatchOption) error {
			if got, ok := obj.(*nvidiacomv1alpha1.PodSnapshotContent); ok && got.Status.Source != nil && failuresRemaining > 0 {
				failuresRemaining--
				return errors.New("source status rejected")
			}
			return c.SubResource(sub).Patch(ctx, obj, patch, opts...)
		},
	}
	w := makeNodeControllerWithInterceptor(t, &fakeCheckpointer{}, funcs, content, pod)
	checkpointPath := filepath.Join(w.config.Storage.BasePath, "Bad_ID", "versions", "1")
	require.NoError(t, os.MkdirAll(checkpointPath, 0o755))
	require.NoError(t, snapshottypes.WriteManifest(checkpointPath, &snapshottypes.CheckpointManifest{
		CheckpointID: "checkpoint-1",
		CreatedAt:    time.Now().UTC(),
		K8s:          snapshottypes.SourcePodManifest{SourceNode: "node-a"},
	}))

	w.reconcilePodSnapshotContent(context.Background(), content.Name)
	afterFailure := getContent(t, w, content.Name)
	assert.True(t, meta.IsStatusConditionTrue(afterFailure.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
	assert.Nil(t, afterFailure.Status.Source)

	w.reconcilePodSnapshotContent(context.Background(), content.Name)
	afterRetry := getContent(t, w, content.Name)
	assert.True(t, meta.IsStatusConditionTrue(afterRetry.Status.Conditions, nvidiacomv1alpha1.PodSnapshotConditionReady))
	require.NotNil(t, afterRetry.Status.Source)
	assert.Equal(t, "node-a", afterRetry.Status.Source.Node)
}
