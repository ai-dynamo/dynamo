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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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

func TestStoreFencesLoadedSnapshot(t *testing.T) {
	tests := []struct {
		name   string
		absent bool
	}{
		{name: "stale existing snapshot"},
		{name: "stale absence", absent: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("load the snapshot from which the next mutation will be derived")
			ctx := context.Background()
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			store := NewStore(fake.NewClientBuilder().WithScheme(scheme).Build(), "test", "group", "uid", "membership")
			snapshot := Snapshot{}
			if !test.absent {
				var err error
				snapshot, err = store.Save(ctx, snapshot, map[string]int{"revision": 1})
				require.NoError(t, err)
			}

			t.Log("persist newer evidence and continue from the returned snapshot without a cache read")
			current, err := store.Save(ctx, snapshot, map[string]int{"revision": 2})
			require.NoError(t, err)
			_, err = store.Save(ctx, current, map[string]int{"revision": 3})
			require.NoError(t, err)

			t.Log("reject the stale mutation rather than retrying its payload against newer state")
			_, err = store.Save(ctx, snapshot, map[string]int{"revision": 1})
			if test.absent {
				assert.True(t, apierrors.IsAlreadyExists(err), "%v", err)
			} else {
				assert.True(t, apierrors.IsConflict(err), "%v", err)
			}

			t.Log("retain the newer evidence for the next reconciliation")
			var observed map[string]int
			loaded, err := store.Load(ctx, &observed)
			require.NoError(t, err)
			require.True(t, loaded.Exists())
			assert.Equal(t, 3, observed["revision"])
		})
	}
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
