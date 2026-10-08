//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func engineGroupControllerTestCheckpoint(t *testing.T, ctx context.Context, kubeClient client.Client, group *api.DynamoGraphDeploymentEngineGroup) engineGroupCheckpoint {
	t.Helper()
	checkpoint, snapshot, err := loadEngineGroupCheckpoint(ctx, engineGroupCheckpointStore(&Reconciler{Client: kubeClient}, group), group)
	require.NoError(t, err)
	require.True(t, snapshot.Exists())
	return checkpoint
}

func TestEngineGroupCheckpointRejectsInvalidRecoveryAuthority(t *testing.T) {
	for _, test := range []struct{ name, message string }{
		{"missing", "checkpoint is missing"},
		{"corrupt", "decode journal"},
		{"unknown version", "unsupported checkpoint version"},
		{"wrong profile", "checkpoint profile"},
		{"wrong binding", "different PodClique incarnation"},
		{"wrong owner", "not owned by Engine Group"},
		{"empty state", "no authoritative base topology"},
		{"invalid state", "invalid checkpoint state"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("initialize a healthy world and persist its recovery authority")
			ctx := context.Background()
			backend := newEngineGroupControllerTestBackend(1)
			group := &api.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "authority", Namespace: "test", UID: "group-uid", Generation: 1},
				Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
			}
			scheme := runtime.NewScheme()
			require.NoError(t, api.AddToScheme(scheme))
			require.NoError(t, corev1.AddToScheme(scheme))
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(group).WithObjects(group).Build()
			controller := &Reconciler{Client: kubeClient, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend}}
			req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
			for step := 0; step < 3; step++ {
				_, err := controller.Reconcile(ctx, req)
				require.NoError(t, err)
			}
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			store := engineGroupCheckpointStore(controller, group)
			checkpoint := engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group)
			journal := &corev1.ConfigMap{}
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKey{Namespace: store.Namespace, Name: store.Name}, journal))

			t.Log("remove, corrupt, or mismatch the private checkpoint without changing the scale target")
			switch test.name {
			case "missing":
				require.NoError(t, kubeClient.Delete(ctx, journal))
			case "corrupt":
				journal.Data["state.json"] = "{"
			case "unknown version":
				checkpoint.Version++
			case "wrong profile":
				checkpoint.ProfileFingerprint = "other-profile"
			case "wrong binding":
				group.Annotations = map[string]string{consts.KubeAnnotationDynamoEngineGroupPodCliqueUID: "new-clique"}
				require.NoError(t, kubeClient.Update(ctx, group))
			case "wrong owner":
				journal.OwnerReferences[0].UID = "other-group"
			case "empty state":
				checkpoint.State.Topologies.Snapshots = nil
			case "invalid state":
				checkpoint.State.ControlRevision = -1
			}
			if test.name != "missing" {
				if test.name != "corrupt" {
					encoded, err := json.Marshal(checkpoint)
					require.NoError(t, err)
					journal.Data["state.json"] = string(encoded)
				}
				require.NoError(t, kubeClient.Update(ctx, journal))
			}

			t.Log("fail closed without creating capacity or submitting a competing membership operation")
			_, err := controller.Reconcile(ctx, req)
			require.ErrorContains(t, err, test.message)
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			assert.Nil(t, group.Status.Operation)
			assert.Equal(t, 0, backend.membershipApplyCount())
			assert.Len(t, backend.capacity.Allocations, 1)
			for _, conditionType := range []string{engineGroupConditionTopologyKnown, engineGroupConditionAvailable, engineGroupConditionDegraded} {
				condition := meta.FindStatusCondition(group.Status.Conditions, conditionType)
				require.NotNil(t, condition)
				assert.Equal(t, metav1.ConditionUnknown, condition.Status)
			}
		})
	}
}

func TestEngineGroupCheckpointWriteFailurePrecedesEffects(t *testing.T) {
	for _, initialized := range []bool{false, true} {
		name := "create"
		if initialized {
			name = "update"
		}
		t.Run(name, func(t *testing.T) {
			t.Log("create a world whose desired target requires growth")
			ctx := context.Background()
			group := &api.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "write-failure", Namespace: "test", UID: "group-uid", Generation: 1},
				Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
			}
			scheme := runtime.NewScheme()
			require.NoError(t, api.AddToScheme(scheme))
			require.NoError(t, corev1.AddToScheme(scheme))
			kubeClient := &checkpointTestClient{Client: fake.NewClientBuilder().WithScheme(scheme).
				WithStatusSubresource(group).WithObjects(group).Build()}
			backend := newEngineGroupControllerTestBackend(1)
			controller := &Reconciler{Client: kubeClient, RuntimeProvider: engineGroupControllerTestRuntimeProvider{backend: backend}}
			req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
			initialSteps := 2
			if initialized {
				initialSteps = 3
			}
			for step := 0; step < initialSteps; step++ {
				_, err := controller.Reconcile(ctx, req)
				require.NoError(t, err)
			}
			kubeClient.failCheckpoint = true

			t.Log("reject persistence and prove no unpersisted target is applied on subsequent reconciles")
			for step := 0; step < 2; step++ {
				_, err := controller.Reconcile(ctx, req)
				require.ErrorContains(t, err, "checkpoint write failed")
			}
			require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
			assert.Nil(t, group.Status.Operation)
			assert.Equal(t, 0, backend.membershipApplyCount())
			assert.Len(t, backend.capacity.Allocations, 1)
			if initialized {
				checkpoint := engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group)
				assert.Nil(t, checkpoint.State.Transition)
			}
		})
	}
}

func TestEngineGroupCheckpointSurvivesPublicStatusWriteFailure(t *testing.T) {
	t.Log("initialize a world and arrange a failed public operation-status publication")
	ctx := context.Background()
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "publication", Namespace: "test", UID: "group-uid", Generation: 1},
		Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, api.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	kubeClient := &checkpointTestClient{Client: fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(group).WithObjects(group).Build()}
	backend := newEngineGroupControllerTestBackend(1)
	backend.ambiguousApply = false
	provider := engineGroupControllerTestRuntimeProvider{backend: backend}
	controller := &Reconciler{Client: kubeClient, RuntimeProvider: provider}
	req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
	for step := 0; step < 3; step++ {
		_, err := controller.Reconcile(ctx, req)
		require.NoError(t, err)
	}
	kubeClient.failStatus = true
	_, err := controller.Reconcile(ctx, req)
	require.ErrorContains(t, err, "public status write failed")
	require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
	require.Nil(t, group.Status.Operation)
	checkpoint := engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group)
	require.NotNil(t, checkpoint.Operation)
	operationID := checkpoint.Operation.ID
	assert.Equal(t, 0, backend.membershipApplyCount())

	t.Log("restart from the private checkpoint, preserving operation identity and source generation")
	kubeClient.failStatus = false
	controller = &Reconciler{Client: kubeClient, RuntimeProvider: provider}
	for step := 0; step < 64; step++ {
		_, err := controller.Reconcile(ctx, req)
		require.NoError(t, err)
		require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
		checkpoint = engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group)
		if checkpoint.State.Transition != nil && checkpoint.State.Transition.Outcome == "Completed" {
			break
		}
	}
	require.NotNil(t, group.Status.Operation)
	assert.Equal(t, operationID, group.Status.Operation.ID)
	assert.Equal(t, int64(1), group.Status.Operation.SpecGeneration)
	assert.Equal(t, api.EngineGroupOperationPhaseCommitted, group.Status.Operation.Phase)
	assert.Equal(t, 1, backend.membershipApplyCount())
	assert.Equal(t, int32(2), group.Status.AvailableReplicas)
}

func TestEngineGroupCheckpointDoesNotRecoverFromPublicProjection(t *testing.T) {
	t.Log("initialize a healthy world with a private checkpoint and public observations")
	ctx := context.Background()
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "projection", Namespace: "test", UID: "group-uid", Generation: 1},
		Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, api.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(group).WithObjects(group).Build()
	backend := newEngineGroupControllerTestBackend(1)
	provider := engineGroupControllerTestRuntimeProvider{backend: backend}
	controller := &Reconciler{Client: kubeClient, RuntimeProvider: provider}
	req := reconcile.Request{NamespacedName: client.ObjectKeyFromObject(group)}
	for step := 0; step < 4; step++ {
		_, err := controller.Reconcile(ctx, req)
		require.NoError(t, err)
	}
	require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
	original := engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group)

	t.Log("change the public summary without changing durable targets or real engine membership")
	group.Status.DesiredNativeMembers = []string{"unrelated-rank"}
	group.Status.Operation = &api.EngineGroupOperationStatus{ID: "unrelated-operation"}
	require.NoError(t, kubeClient.Status().Update(ctx, group))

	t.Log("restart and repair the projection from private authority and fresh observations, without effects")
	controller = &Reconciler{Client: kubeClient, RuntimeProvider: provider}
	_, err := controller.Reconcile(ctx, req)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(ctx, req.NamespacedName, group))
	assert.Equal(t, original.DesiredNativeMembers, group.Status.DesiredNativeMembers)
	assert.Nil(t, group.Status.Operation)
	assert.Equal(t, 0, backend.membershipApplyCount())
	assert.Equal(t, original.State, engineGroupControllerTestCheckpoint(t, ctx, kubeClient, group).State)
}

type checkpointTestClient struct {
	client.Client
	failCheckpoint bool
	failStatus     bool
}

func (c *checkpointTestClient) Create(ctx context.Context, object client.Object, opts ...client.CreateOption) error {
	if _, isCheckpoint := object.(*corev1.ConfigMap); isCheckpoint && c.failCheckpoint {
		return errors.New("checkpoint write failed")
	}
	return c.Client.Create(ctx, object, opts...)
}

func (c *checkpointTestClient) Update(ctx context.Context, object client.Object, opts ...client.UpdateOption) error {
	if _, isCheckpoint := object.(*corev1.ConfigMap); isCheckpoint && c.failCheckpoint {
		return errors.New("checkpoint write failed")
	}
	return c.Client.Update(ctx, object, opts...)
}

func (c *checkpointTestClient) Status() client.SubResourceWriter {
	return checkpointTestStatusWriter{SubResourceWriter: c.Client.Status(), owner: c}
}

type checkpointTestStatusWriter struct {
	client.SubResourceWriter
	owner *checkpointTestClient
}

func (w checkpointTestStatusWriter) Update(ctx context.Context, object client.Object, opts ...client.SubResourceUpdateOption) error {
	if w.owner.failStatus {
		return errors.New("public status write failed")
	}
	return w.SubResourceWriter.Update(ctx, object, opts...)
}
