//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestPowerCapReplicaSetOwnershipUsesMetadataCache(t *testing.T) {
	t.Log("Seed an owner with an engine template that ownership observation does not need")
	env := sharedEnv.ForTest(t)
	dgd := powerCapTestDGD(t, "component")
	dgd.Namespace = env.Namespace()
	rs := &appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{Name: "power-owner", Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, dgd.GroupVersionKind())}}, Spec: appsv1.ReplicaSetSpec{
		Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"worker": "power"}},
		Template: corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"worker": "power"}}, Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "engine", Image: "example/engine:test"}}}},
	}}
	rs.OwnerReferences[0].APIVersion, rs.OwnerReferences[0].Kind = "nvidia.com/v1beta1", "DynamoGraphDeployment"
	require.NoError(t, env.Client().Create(t.Context(), rs))
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(rs, appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))}}}

	t.Log("Register only a metadata informer and forbid lazy creation of typed informers")
	cached, err := cache.New(env.RESTConfig(), cache.Options{Scheme: env.Client().Scheme(), ReaderFailOnMissingInformer: true, DefaultNamespaces: map[string]cache.Config{env.Namespace(): {}}})
	require.NoError(t, err)
	metadata := &metav1.PartialObjectMetadata{}
	metadata.SetGroupVersionKind(appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))
	_, err = cached.GetInformer(t.Context(), metadata)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { done <- cached.Start(ctx) }()
	t.Cleanup(func() {
		cancel()
		require.NoError(t, <-done)
	})
	syncCtx, syncCancel := context.WithTimeout(ctx, 10*time.Second)
	defer syncCancel()
	require.True(t, cached.WaitForCacheSync(syncCtx))

	t.Log("Ownership resolves through metadata while a typed ReplicaSet read is explicitly uncached")
	owned, pending, err := powerCapPodOwned(ctx, cached, dgd, pod)
	require.NoError(t, err)
	require.True(t, owned)
	require.False(t, pending)
	err = cached.Get(ctx, client.ObjectKeyFromObject(rs), &appsv1.ReplicaSet{})
	require.ErrorContains(t, err, "not cached")
	pod.OwnerReferences[0].UID = "stale-owner-uid"
	owned, pending, err = powerCapPodOwned(ctx, cached, dgd, pod)
	require.NoError(t, err)
	require.False(t, owned)
	require.False(t, pending)
}
