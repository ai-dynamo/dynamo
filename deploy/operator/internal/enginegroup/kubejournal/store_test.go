/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package kubejournal

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestObjectName(t *testing.T) {
	t.Log("preserve a short Engine Group name so operators can recognize its journal")
	assert.Equal(t, "group-a-membership", objectName("group-a", "membership"))

	t.Log("bound long generated names while retaining a stable suffix")
	name := objectName("an-engine-group-name-that-is-deliberately-longer-than-a-kubernetes-object-name-allows", "membership")
	assert.LessOrEqual(t, len(name), 63)
	assert.Contains(t, name, "membership")
	assert.Equal(t, name, objectName("an-engine-group-name-that-is-deliberately-longer-than-a-kubernetes-object-name-allows", "membership"))
}

func TestStoreRejectsJournalFromPreviousGroupIncarnation(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	stale := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{
		Namespace: "test",
		Name:      "group-membership",
		OwnerReferences: []metav1.OwnerReference{{
			APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeploymentEngineGroup",
			Name: "group", UID: types.UID("old-uid"),
		}},
	}, Data: map[string]string{stateDataKey: `{}`}}
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(stale).Build()
	store := NewStore(kubeClient, "test", "group", types.UID("new-uid"), "membership")

	t.Log("refuse stale operation evidence after an Engine Group name is reused")
	var state map[string]any
	_, err := store.Load(context.Background(), &state)
	assert.ErrorContains(t, err, "new-uid")
}
