/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	enginegroupcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDGDEngineGroupDeletionIsDeferred(t *testing.T) {
	t.Log("give an owned world a controller finalizer while leaving another graph's world untouched")
	dgd := &api.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: "dgd-uid"}}
	group := &api.DynamoGraphDeploymentEngineGroup{ObjectMeta: metav1.ObjectMeta{
		Name: "world", Namespace: dgd.Namespace, UID: "owned-world-uid", Finalizers: []string{"test-retirement"},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, api.GroupVersion.WithKind("DynamoGraphDeployment"))},
	}}
	unrelated := &api.DynamoGraphDeploymentEngineGroup{ObjectMeta: metav1.ObjectMeta{Name: "unrelated", Namespace: dgd.Namespace, UID: "unrelated-uid"}}
	kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(group, unrelated).Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: kube}

	t.Log("request child deletion and keep the parent until engine retirement finishes")
	require.ErrorIs(t, reconciler.FinalizeResource(t.Context(), dgd), errEngineGroupRetirementPending)
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(group), group))
	require.NotNil(t, group.DeletionTimestamp)
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(unrelated), unrelated))
	assert.Nil(t, unrelated.DeletionTimestamp)

	t.Log("repeated parent reconciliation waits without rewriting the child's scale target")
	require.ErrorIs(t, reconciler.FinalizeResource(t.Context(), dgd), errEngineGroupRetirementPending)

	t.Log("when the child completes, the owned-world handoff no longer blocks the parent")
	group.Finalizers = nil
	require.NoError(t, kube.Update(t.Context(), group))
	pending, err := (&enginegroupcontroller.GroveChildrenReconciler{Client: kube}).RequestDeletion(t.Context(), dgd)
	require.NoError(t, err)
	assert.False(t, pending)
}
