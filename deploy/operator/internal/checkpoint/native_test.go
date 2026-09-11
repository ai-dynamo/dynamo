/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"context"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestResolvePodSnapshotForService(t *testing.T) {
	t.Run("resolves a Ready compatible PodSnapshot", func(t *testing.T) {
		t.Log("Given a Ready PodSnapshot with matching Dynamo compatibility metadata")
		snapshot := nativeTestPodSnapshot()
		reader := fake.NewClientBuilder().WithScheme(nativeTestScheme(t)).WithObjects(snapshot).Build()
		config := nativeTestCheckpointConfig(snapshot.Name)

		t.Log("When the explicit reference is resolved for the same worker generation")
		info, err := ResolvePodSnapshotForService(
			context.Background(),
			reader,
			snapshot.Namespace,
			config,
			ptr.To("worker-v1"),
			ExplicitPodSnapshotUse(),
		)

		t.Log("Then the resolver returns the bound native artifact identity and source container")
		require.NoError(t, err)
		require.NotNil(t, info.NativeSnapshot)
		assert.True(t, info.Enabled)
		assert.True(t, info.Exists)
		assert.True(t, info.Ready)
		assert.Equal(t, snapshot.Name, info.CheckpointName)
		assert.Equal(t, snapshot.UID, info.NativeSnapshot.UID)
		assert.Equal(t, "content-a", info.NativeSnapshot.BoundContentName)
		assert.Equal(t, "main", info.NativeSnapshot.SourceContainer)
		assert.Nil(t, info.GPUMemoryService)
		assert.Equal(t, []string{"engine-0"}, info.RestoreTargetContainers)
		assert.Equal(t, nvidiacomv1alpha1.CheckpointStartupPolicyImmediate, info.StartupPolicy)
	})

	t.Run("returns a compatible pending snapshot for workload gating", func(t *testing.T) {
		t.Log("Given a compatible PodSnapshot whose capture has not completed")
		snapshot := nativeTestPodSnapshot()
		snapshot.Status = snapshotv1alpha1.PodSnapshotStatus{}
		reader := fake.NewClientBuilder().WithScheme(nativeTestScheme(t)).WithObjects(snapshot).Build()

		t.Log("When the reference is resolved")
		info, err := ResolvePodSnapshotForService(
			context.Background(),
			reader,
			snapshot.Namespace,
			nativeTestCheckpointConfig(snapshot.Name),
			ptr.To("worker-v1"),
			ExplicitPodSnapshotUse(),
		)

		t.Log("Then reconciliation can retain the UID while keeping the workload gated")
		require.NoError(t, err)
		assert.False(t, info.Ready)
		require.NotNil(t, info.NativeSnapshot)
		assert.Equal(t, snapshot.UID, info.NativeSnapshot.UID)
		assert.Empty(t, info.NativeSnapshot.BoundContentName)
	})

	t.Run("restores the declared GMS topology", func(t *testing.T) {
		tests := []struct {
			name string
			mode string
			want nvidiacomv1alpha1.GPUMemoryServiceMode
		}{
			{name: "intra-pod", mode: string(nvidiacomv1alpha1.GMSModeIntraPod), want: nvidiacomv1alpha1.GMSModeIntraPod},
			{name: "inter-pod", mode: string(nvidiacomv1alpha1.GMSModeInterPod), want: nvidiacomv1alpha1.GMSModeInterPod},
		}

		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				t.Log("Given a compatible snapshot declaring an enabled GMS topology")
				snapshot := nativeTestPodSnapshot()
				snapshot.Annotations[consts.SnapshotGMSModeAnnotation] = test.mode
				reader := fake.NewClientBuilder().WithScheme(nativeTestScheme(t)).WithObjects(snapshot).Build()

				t.Log("When the native snapshot is resolved")
				info, err := ResolvePodSnapshotForService(
					context.Background(),
					reader,
					snapshot.Namespace,
					nativeTestCheckpointConfig(snapshot.Name),
					ptr.To("worker-v1"),
					ExplicitPodSnapshotUse(),
				)

				t.Log("Then Dynamo reconstructs only the compatibility topology for later client overlay")
				require.NoError(t, err)
				require.NotNil(t, info.GPUMemoryService)
				assert.True(t, info.GPUMemoryService.Enabled)
				assert.Equal(t, test.want, info.GPUMemoryService.Mode)
			})
		}
	})

	t.Run("resolves a non-worker snapshot without a worker hash", func(t *testing.T) {
		t.Log("Given a compatible non-worker PodSnapshot with no worker generation")
		snapshot := nativeTestPodSnapshot()
		delete(snapshot.Annotations, consts.SnapshotWorkerHashAnnotation)
		reader := fake.NewClientBuilder().WithScheme(nativeTestScheme(t)).WithObjects(snapshot).Build()

		t.Log("When the reference is resolved without a worker hash contract")
		info, err := ResolvePodSnapshotForService(
			context.Background(),
			reader,
			snapshot.Namespace,
			nativeTestCheckpointConfig(snapshot.Name),
			nil,
			ExplicitPodSnapshotUse(),
		)

		t.Log("Then the remaining native compatibility contract is still enforced")
		require.NoError(t, err)
		assert.True(t, info.Ready)
		require.NotNil(t, info.NativeSnapshot)
		assert.Equal(t, snapshot.UID, info.NativeSnapshot.UID)
	})

	t.Run("rejects a worker restore before its hash is available", func(t *testing.T) {
		t.Log("Given a worker restore whose generation identity is not initialized")
		snapshot := nativeTestPodSnapshot()
		reader := fake.NewClientBuilder().WithScheme(nativeTestScheme(t)).WithObjects(snapshot).Build()

		t.Log("When resolution requires an empty worker hash")
		_, err := ResolvePodSnapshotForService(
			context.Background(),
			reader,
			snapshot.Namespace,
			nativeTestCheckpointConfig(snapshot.Name),
			ptr.To(""),
			ExplicitPodSnapshotUse(),
		)

		t.Log("Then resolution fails closed until the worker generation is known")
		require.ErrorContains(t, err, "worker compatibility hash is required")
	})
}

func TestResolvePodSnapshotForServiceRejectsIncompatibleReferences(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*snapshotv1alpha1.PodSnapshot)
		wantErr string
	}{
		{
			name: "missing UID",
			mutate: func(snapshot *snapshotv1alpha1.PodSnapshot) {
				snapshot.UID = ""
			},
			wantErr: "has no UID",
		},
		{
			name: "failed capture",
			mutate: func(snapshot *snapshotv1alpha1.PodSnapshot) {
				snapshot.Status.Conditions = []metav1.Condition{{
					Type:   snapshotv1alpha1.PodSnapshotConditionFailed,
					Status: metav1.ConditionTrue,
				}}
			},
			wantErr: "has failed",
		},
		{
			name: "ambiguous source",
			mutate: func(snapshot *snapshotv1alpha1.PodSnapshot) {
				snapshot.Spec.Source.PodRef.Containers = []string{"main", "other"}
			},
			wantErr: "exactly one source container",
		},
		{
			name: "unsupported compatibility version",
			mutate: func(snapshot *snapshotv1alpha1.PodSnapshot) {
				snapshot.Annotations[consts.SnapshotCompatibilityVersionAnnotation] = "v2"
			},
			wantErr: "unsupported Dynamo compatibility version",
		},
		{
			name: "worker generation mismatch",
			mutate: func(snapshot *snapshotv1alpha1.PodSnapshot) {
				snapshot.Annotations[consts.SnapshotWorkerHashAnnotation] = "worker-v2"
			},
			wantErr: "does not match expected hash",
		},
		{
			name: "unsupported GMS mode",
			mutate: func(snapshot *snapshotv1alpha1.PodSnapshot) {
				snapshot.Annotations[consts.SnapshotGMSModeAnnotation] = "external"
			},
			wantErr: "GMS mode",
		},
		{
			name: "Ready without bound content",
			mutate: func(snapshot *snapshotv1alpha1.PodSnapshot) {
				snapshot.Status.BoundPodSnapshotContentName = nil
			},
			wantErr: "has no bound PodSnapshotContent",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Given a PodSnapshot that violates one native restore invariant")
			snapshot := nativeTestPodSnapshot()
			test.mutate(snapshot)
			reader := fake.NewClientBuilder().WithScheme(nativeTestScheme(t)).WithObjects(snapshot).Build()

			t.Log("When the explicit reference is resolved")
			_, err := ResolvePodSnapshotForService(
				context.Background(),
				reader,
				snapshot.Namespace,
				nativeTestCheckpointConfig(snapshot.Name),
				ptr.To("worker-v1"),
				ExplicitPodSnapshotUse(),
			)

			t.Log("Then the resolver fails closed with the violated invariant")
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

func TestResolvePodSnapshotForServiceRetainedAutomaticCheckpoint(t *testing.T) {
	t.Log("Given a compatible automatic PodSnapshot configured for retention")
	ownerUID := types.UID("owning-dgd-uid")
	snapshot := nativeTestPodSnapshot()
	snapshot.Annotations[consts.CheckpointAutoAnnotation] = consts.KubeLabelValueTrue
	snapshot.Annotations[consts.CheckpointDeletionPolicyAnnotation] = string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain)
	snapshot.Annotations[consts.CheckpointOwnerUIDAnnotation] = string(ownerUID)
	reader := fake.NewClientBuilder().WithScheme(nativeTestScheme(t)).WithObjects(snapshot).Build()
	config := nativeTestCheckpointConfig(snapshot.Name)

	t.Run("explicit checkpointRef is rejected", func(t *testing.T) {
		t.Log("When the retained automatic snapshot is resolved as a public checkpointRef")
		_, err := ResolvePodSnapshotForService(
			context.Background(),
			reader,
			snapshot.Namespace,
			config,
			ptr.To("worker-v1"),
			ExplicitPodSnapshotUse(),
		)

		t.Log("Then the resolver rejects adoption of the retained artifact")
		require.ErrorContains(t, err, "retained automatic checkpoint")
	})

	t.Run("owning DGD managed restore remains valid", func(t *testing.T) {
		t.Log("When the snapshot is resolved for the DGD incarnation that created it")
		info, err := ResolvePodSnapshotForService(
			context.Background(),
			reader,
			snapshot.Namespace,
			config,
			ptr.To("worker-v1"),
			ManagedPodSnapshotUse(ownerUID),
		)

		t.Log("Then the managed restore remains ready")
		require.NoError(t, err)
		assert.True(t, info.Ready)
	})

	t.Run("different DGD cannot adopt the retained checkpoint", func(t *testing.T) {
		t.Log("When another DGD incarnation resolves the retained automatic snapshot")
		_, err := ResolvePodSnapshotForService(
			context.Background(),
			reader,
			snapshot.Namespace,
			config,
			ptr.To("worker-v1"),
			ManagedPodSnapshotUse("different-dgd-uid"),
		)

		t.Log("Then the resolver rejects the mismatched owner UID")
		require.ErrorContains(t, err, "belongs to DGD uid")
	})
}

func nativeTestPodSnapshot() *snapshotv1alpha1.PodSnapshot {
	return &snapshotv1alpha1.PodSnapshot{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "snapshot-a",
			Namespace: "test-ns",
			UID:       types.UID("snapshot-uid"),
			Annotations: map[string]string{
				consts.SnapshotCompatibilityVersionAnnotation: consts.SnapshotCompatibilityVersion,
				consts.SnapshotWorkerHashAnnotation:           "worker-v1",
				consts.SnapshotGMSModeAnnotation:              consts.SnapshotGMSModeDisabled,
			},
		},
		Spec: snapshotv1alpha1.PodSnapshotSpec{
			Source: snapshotv1alpha1.PodSnapshotSource{
				PodRef: snapshotv1alpha1.PodReference{
					Name:       "capture-pod",
					Containers: []string{"main"},
				},
			},
		},
		Status: snapshotv1alpha1.PodSnapshotStatus{
			BoundPodSnapshotContentName: ptr.To("content-a"),
			Conditions: []metav1.Condition{{
				Type:   snapshotv1alpha1.PodSnapshotConditionReady,
				Status: metav1.ConditionTrue,
			}},
		},
	}
}

func nativeTestCheckpointConfig(name string) *nvidiacomv1alpha1.ServiceCheckpointConfig {
	return &nvidiacomv1alpha1.ServiceCheckpointConfig{
		Enabled:             true,
		CheckpointRef:       ptr.To(name),
		TargetContainerName: "engine-0",
	}
}

func nativeTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, snapshotv1alpha1.AddToScheme(scheme))
	return scheme
}
