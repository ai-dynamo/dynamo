//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestServiceIdentityBackfillPreservesServiceContract(t *testing.T) {
	t.Log("Create a real Grove component Service and retain its allocated network identity")
	env := sharedEnv.ForTest(t)
	ctx, kubeClient := t.Context(), env.Client()
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: "service-identity", Namespace: env.Namespace(), UID: "graph-uid",
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
			}},
		},
	}
	component := &dgd.Spec.Components[0]
	r := newGroveStableResourcesReconciler(kubeClient, events.NewFakeRecorder(20), &configv1alpha1.OperatorConfiguration{})
	_, err := r.reconcileComponentService(ctx, dgd, component, nil, true)
	require.NoError(t, err)
	key := client.ObjectKey{Namespace: dgd.Namespace, Name: dgd.Name + "-frontend"}
	service := &corev1.Service{}
	require.NoError(t, kubeClient.Get(ctx, key, service))
	original := service.DeepCopy()

	t.Log("Simulate an existing pre-upgrade Service with external metadata and no identity labels")
	service.Labels = map[string]string{"example.com/team": "external"}
	service.Annotations = map[string]string{
		"example.com/note":                             "external",
		commoncontroller.NvidiaAnnotationHashKey:       original.Annotations[commoncontroller.NvidiaAnnotationHashKey],
		commoncontroller.NvidiaAnnotationGenerationKey: original.Annotations[commoncontroller.NvidiaAnnotationGenerationKey],
	}
	require.NoError(t, kubeClient.Update(ctx, service, client.FieldOwner("external-service-manager")))
	_, err = r.reconcileComponentService(ctx, dgd, component, nil, true)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(ctx, key, service))
	require.Equal(t, original.UID, service.UID)
	require.Equal(t, original.Spec, service.Spec)
	require.Equal(t, original.OwnerReferences, service.OwnerReferences)
	require.Equal(t, original.Annotations[commoncontroller.NvidiaAnnotationHashKey], service.Annotations[commoncontroller.NvidiaAnnotationHashKey])
	require.Equal(t, original.Annotations[commoncontroller.NvidiaAnnotationGenerationKey], service.Annotations[commoncontroller.NvidiaAnnotationGenerationKey])
	require.Equal(t, "external", service.Labels["example.com/team"])
	require.Equal(t, "external", service.Annotations["example.com/note"])
	require.Equal(t, dgd.Name, service.Labels[consts.KubeLabelDynamoGraphDeploymentName])
	for key, value := range original.Spec.Selector {
		require.Equal(t, value, service.Labels[key], "Service identity must match the unchanged Pod selector")
	}

	t.Log("Repair a conflicting identity label and then reconcile without another write")
	service.Labels[consts.KubeLabelDynamoComponent] = "wrong"
	require.NoError(t, kubeClient.Update(ctx, service, client.FieldOwner("external-service-manager")))
	_, err = r.reconcileComponentService(ctx, dgd, component, nil, true)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(ctx, key, service))
	require.Equal(t, component.ComponentName, service.Labels[consts.KubeLabelDynamoComponent])
	version := service.ResourceVersion
	_, err = r.reconcileComponentService(ctx, dgd, component, nil, true)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(ctx, key, service))
	require.Equal(t, version, service.ResourceVersion)
}
