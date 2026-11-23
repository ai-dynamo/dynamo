//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"testing"

	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func TestSyncResourceRemovesInitiallyCreatedContent(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		name := "SSA creation"
		if legacy {
			name = "legacy Update ownership"
		}
		t.Run(name, func(t *testing.T) {
			env := sharedEnv.ForTest(t)
			ctx, kubeClient := t.Context(), env.Client()
			r := newDGDResourceSyncer(kubeClient, nil)

			t.Log("Create an owned ConfigMap with an unchanged field to remove later")
			parent := &corev1.ConfigMap{Name: "parent", Namespace: env.Namespace()}
			require.NoError(t, kubeClient.Create(ctx, parent))
			desired := &corev1.ConfigMap{
				Name: "config", Namespace: env.Namespace(),
				Data: map[string]string{"keep": "value", "remove": "initial-value"},
			}
			if legacy {
				created := desired.DeepCopy()
				require.NoError(t, controllerutil.SetControllerReference(parent, created, kubeClient.Scheme()))
				hash, err := commoncontroller.GetSpecHash(created)
				require.NoError(t, err)
				created.Annotations = map[string]string{
					commoncontroller.NvidiaAnnotationHashKey:       hash,
					commoncontroller.NvidiaAnnotationGenerationKey: "1",
				}
				require.NoError(t, kubeClient.Create(ctx, created, client.FieldOwner("legacy-operator")))
			} else {
				_, _, err := commoncontroller.SyncResource(ctx, &r, parent, func(context.Context) (*corev1.ConfigMap, bool, error) {
					return desired, false, nil
				})
				require.NoError(t, err)
			}

			t.Log("A lifecycle controller adds a finalizer under the legacy shared field-manager name")
			stored := &corev1.ConfigMap{}
			key := client.ObjectKeyFromObject(desired)
			require.NoError(t, kubeClient.Get(ctx, key, stored))
			stored.Finalizers = []string{"example.com/lifecycle"}
			manager := "dynamo-operator"
			if legacy {
				manager = "legacy-operator"
			}
			require.NoError(t, kubeClient.Update(ctx, stored, client.FieldOwner(manager)))

			t.Log("An independent manager adds a key outside the operator's rendered intent")
			external := stored.DeepCopy()
			external.Data["foreign"] = "preserved"
			require.NoError(t, kubeClient.Patch(ctx, external, client.MergeFrom(stored), client.FieldOwner("external-manager")))

			t.Log("Remove the unchanged initial field without claiming the independent manager's key")
			delete(desired.Data, "remove")
			_, _, err := commoncontroller.SyncResource(ctx, &r, parent, func(context.Context) (*corev1.ConfigMap, bool, error) {
				return desired, false, nil
			})
			require.NoError(t, err)
			require.NoError(t, kubeClient.Get(ctx, key, stored))
			require.Equal(t, map[string]string{"keep": "value", "foreign": "preserved"}, stored.Data)
			require.Equal(t, []string{"example.com/lifecycle"}, stored.Finalizers)

			t.Log("The lifecycle controller can release its finalizer without synchronization restoring it")
			stored.Finalizers = nil
			require.NoError(t, kubeClient.Update(ctx, stored, client.FieldOwner("lifecycle-controller")))
			_, _, err = commoncontroller.SyncResource(ctx, &r, parent, func(context.Context) (*corev1.ConfigMap, bool, error) {
				return desired, false, nil
			})
			require.NoError(t, err)
			require.NoError(t, kubeClient.Get(ctx, key, stored))
			require.Empty(t, stored.Finalizers)
		})
	}
}
