/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoPoolRelayDependencyIndexMapsOnlySelectedGraphs(t *testing.T) {
	t.Log("Index a missing transport Secret before it exists")
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, v1alpha1.AddToScheme(scheme))
	require.NoError(t, v1beta1.AddToScheme(scheme))
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "workers"}, Spec: v1beta1.DynamoGraphDeploymentSpec{Env: []corev1.EnvVar{{Name: "NATS_AUTH_TOKEN", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "auth"}, Key: "token"}}}}}}
	selected := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "selected", Namespace: "workers"}, Spec: v1alpha1.DynamoPoolRelaySpec{Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{Refs: []v1alpha1.DynamoPoolRelayDGDRef{{Name: "model"}}}}}}
	other := &v1alpha1.DynamoPoolRelay{ObjectMeta: metav1.ObjectMeta{Name: "other", Namespace: "workers"}, Spec: v1alpha1.DynamoPoolRelaySpec{Sources: v1alpha1.DynamoPoolRelaySourcesSpec{DGDs: v1alpha1.DynamoPoolRelayDGDSelection{Refs: []v1alpha1.DynamoPoolRelayDGDRef{{Name: "unrelated"}}}}}}
	c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithIndex(&v1beta1.DynamoGraphDeployment{}, relayDependencyIndex, relayDGDDependencies).WithObjects(dgd, selected, other).Build()
	reconciler := &relayReconciler{Client: c}
	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "auth", Namespace: "workers"}}
	requests := reconciler.relaysForDependency(context.Background(), secret)
	require.Len(t, requests, 1)
	require.Equal(t, client.ObjectKeyFromObject(selected), requests[0].NamespacedName)

	t.Log("An unrelated Secret update does not requeue those Relays")
	secret.Name = "unrelated"
	require.Empty(t, reconciler.relaysForDependency(context.Background(), secret))
}
