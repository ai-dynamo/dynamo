//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
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
	rs := &appsv1.ReplicaSet{Name: "power-owner", Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, dgd.GroupVersionKind())}, Spec: appsv1.ReplicaSetSpec{
		Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"worker": "power"}},
		Template: corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"worker": "power"}}, Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "engine", Image: "example/engine:test"}}}},
	}}
	rs.OwnerReferences[0].APIVersion, rs.OwnerReferences[0].Kind = "nvidia.com/v1beta1", "DynamoGraphDeployment"
	require.NoError(t, env.Client().Create(t.Context(), rs))
	pod := &corev1.Pod{Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(rs, appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))}}

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

func TestPowerCapComponentPreservationIgnoresAPIServerDefaults(t *testing.T) {
	t.Log("Store one 300 W worker DCD through the API server and admission webhooks")
	env := sharedEnv.ForTest(t)
	c := env.Client()
	dgd := powerCapTestDGD(t, "component")
	dgd.Namespace, dgd.UID = env.Namespace(), "power-dgd-uid"
	dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image = "example/engine:1.6.0"
	hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = hash
	generated, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: hash})
	require.NoError(t, err)
	dcd := generated["worker"]
	dcd.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
	require.NoError(t, c.Create(t.Context(), dcd))

	t.Log("Store its Deployment with a Kubernetes-discovery service account the API server mirrors")
	config := &configv1alpha1.OperatorConfiguration{}
	config.Discovery.Backend = configv1alpha1.DiscoveryBackendKubernetes
	renderer := newDCDWorkloadRenderer(c, config, &commoncontroller.RuntimeConfig{}, &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }})
	template, err := renderer.generatePodTemplateSpec(t.Context(), dcd, dynamo.RoleMain, renderer.containerGPUCount(t.Context(), dcd))
	require.NoError(t, err)
	require.NotEmpty(t, template.Spec.ServiceAccountName)
	require.Empty(t, template.Spec.DeprecatedServiceAccount)
	deployment := &appsv1.Deployment{Name: dcd.Name, Namespace: dcd.Namespace, OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}, Spec: appsv1.DeploymentSpec{Selector: &metav1.LabelSelector{MatchLabels: template.Labels}, Template: *template}}
	require.NoError(t, c.Create(t.Context(), deployment))
	require.Equal(t, template.Spec.ServiceAccountName, deployment.Spec.Template.Spec.DeprecatedServiceAccount)

	t.Log("A cap-only edit keeps the observed generation and its creation cap")
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	workloads := newComponentWorkloadsReconciler(c, nil, newDGDWorkerRolloutReconciler(c, nil))
	retained, err := workloads.resolvePowerCapPreservation(t.Context(), dgd, nil)
	require.NoError(t, err)
	require.NotNil(t, retained)
	require.Equal(t, hash, retained.hash)
	require.Equal(t, "300", retained.caps["worker"])
}
