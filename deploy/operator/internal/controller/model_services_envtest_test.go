//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/modelservice"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestGraphModelServicesOwnershipRepairAndCleanup(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx, kubeClient := t.Context(), env.Client()
	reconciler := modelservice.Reconciler{Client: kubeClient}

	t.Log("Create two graphs serving the same model and a second model in the first graph")
	first := &v1beta1.DynamoGraphDeployment{Name: "first", Namespace: env.Namespace()}
	second := &v1beta1.DynamoGraphDeployment{Name: "second", Namespace: env.Namespace()}
	first.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{
		{ComponentName: "prefill", ModelRef: &v1beta1.ModelReference{Name: "llama-3"}},
		{ComponentName: "decode", ModelRef: &v1beta1.ModelReference{Name: "llama-3"}},
		{ComponentName: "other", ModelRef: &v1beta1.ModelReference{Name: "other-model"}},
	}
	for i := range first.Spec.Components {
		component := &first.Spec.Components[i]
		component.ComponentType = v1beta1.ComponentTypeWorker
		component.RuntimeVersionOverride = "1.7.0"
		component.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "main", Image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.7.0",
			Command: []string{"python3", "-m", "dynamo.vllm"}, Args: []string{"--model", component.ModelRef.Name},
		}}}}
	}
	first.Spec.Annotations = map[string]string{"example.com/remove": "first"}
	second.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{*first.Spec.Components[0].DeepCopy()}
	require.NoError(t, kubeClient.Create(ctx, first))
	require.NoError(t, kubeClient.Create(ctx, second))
	firstNames, err := reconciler.Reconcile(ctx, first)
	require.NoError(t, err)
	require.Len(t, firstNames, 2)
	secondNames, err := reconciler.Reconcile(ctx, second)
	require.NoError(t, err)
	require.Len(t, secondNames, 1)
	key := client.ObjectKey{Namespace: env.Namespace(), Name: modelservice.Name("llama-3", first.Name, true)}
	service := &corev1.Service{}
	require.NoError(t, kubeClient.Get(ctx, key, service))
	require.True(t, metav1.IsControlledBy(service, first))
	require.Equal(t, first.Name, service.Spec.Selector[consts.KubeLabelDynamoGraphDeploymentName])
	require.NotContains(t, secondNames, service.Name)

	t.Log("Repair metadata and selector drift while preserving independently owned lifecycle fields")
	service.Labels[modelservice.LabelModelDiscovery] = "wrong"
	service.Spec.Selector[consts.KubeLabelDynamoGraphDeploymentName] = second.Name
	service.Finalizers = []string{"example.com/external"}
	service.Annotations["example.com/external"] = "keep"
	require.NoError(t, kubeClient.Update(ctx, service, client.FieldOwner("external-controller")))
	first.Spec.Annotations = nil
	_, err = reconciler.Reconcile(ctx, first)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(ctx, key, service))
	require.Equal(t, "true", service.Labels[modelservice.LabelModelDiscovery])
	require.Equal(t, first.Name, service.Spec.Selector[consts.KubeLabelDynamoGraphDeploymentName])
	require.NotContains(t, service.Annotations, "example.com/remove")
	require.Equal(t, "keep", service.Annotations["example.com/external"])
	require.Equal(t, []string{"example.com/external"}, service.Finalizers)
	version := service.ResourceVersion
	_, err = reconciler.Reconcile(ctx, first)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(ctx, key, service))
	require.Equal(t, version, service.ResourceVersion)

	t.Log("Remove a model and preserve another graph and unrelated graph-owned Services")
	unrelated := &corev1.Service{Name: "frontend", Namespace: env.Namespace(), OwnerReferences: service.OwnerReferences,
		Spec: corev1.ServiceSpec{Ports: []corev1.ServicePort{{Port: 8000}}}}
	require.NoError(t, kubeClient.Create(ctx, unrelated))
	removed := client.ObjectKey{Namespace: env.Namespace(), Name: modelservice.Name("other-model", first.Name, true)}
	stale := &corev1.Service{}
	require.NoError(t, kubeClient.Get(ctx, removed, stale))
	delete(stale.Labels, modelservice.LabelModelDiscovery)
	require.NoError(t, kubeClient.Update(ctx, stale, client.FieldOwner("external-controller")))
	first.Spec.Components = first.Spec.Components[:2]
	remainingNames, err := reconciler.Reconcile(ctx, first)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(ctx, removed, &corev1.Service{}), "retirement must wait for the provider cutover")
	require.NoError(t, reconciler.DeleteStale(ctx, first, remainingNames))
	require.True(t, apierrors.IsNotFound(kubeClient.Get(ctx, removed, &corev1.Service{})))
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(unrelated), &corev1.Service{}))
	for name := range secondNames {
		require.NoError(t, kubeClient.Get(ctx, client.ObjectKey{Namespace: env.Namespace(), Name: name}, &corev1.Service{}))
	}

	t.Log("Release the independent finalizer before namespace cleanup")
	service.Finalizers = nil
	require.NoError(t, kubeClient.Update(ctx, service, client.FieldOwner("external-controller")))
}

func TestDynamoModelFindsGraphNamedModelServiceEndpoints(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx, kubeClient := t.Context(), env.Client()

	t.Log("Represent Kubernetes-generated EndpointSlices for two graph-scoped Services")
	modelHash := modelservice.HashModelName("llama-3")
	for i, graph := range []string{"first", "second"} {
		slice := &discoveryv1.EndpointSlice{
			Name: graph + "-slice", Namespace: env.Namespace(),
			Labels: map[string]string{consts.KubeLabelDynamoBaseModelHash: modelHash,
				discoveryv1.LabelServiceName: modelservice.Name("llama-3", graph, true)},
			AddressType: discoveryv1.AddressTypeIPv4,
			Endpoints: []discoveryv1.Endpoint{{Addresses: []string{fmt.Sprintf("10.0.0.%d", i+1)},
				TargetRef: &corev1.ObjectReference{Kind: "Pod", Name: graph + "-worker"}}},
		}
		require.NoError(t, kubeClient.Create(ctx, slice))
	}

	t.Log("The existing consumer discovers both graphs through the unchanged model label")
	r := &DynamoModelReconciler{Client: kubeClient}
	model := &nvidiacomv1alpha1.DynamoModel{Namespace: env.Namespace(), Spec: nvidiacomv1alpha1.DynamoModelSpec{BaseModelName: "llama-3", ModelType: "base"}}
	candidates, names, err := r.getEndpointCandidates(ctx, model)
	require.NoError(t, err)
	require.Len(t, candidates, 2)
	require.Len(t, names, 2)
	require.Contains(t, names, modelservice.Name("llama-3", "first", true))
	require.Contains(t, names, modelservice.Name("llama-3", "second", true))
}
