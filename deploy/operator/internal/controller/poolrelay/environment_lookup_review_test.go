// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package poolrelay

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestRelayKeepsKnownDiscoveryAliasWithUnrelatedPendingBindings(t *testing.T) {
	t.Log("A known discovery alias remains usable while final topology ownership is pending")
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Env: []corev1.EnvVar{{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "$(Z_MODE)"}, {Name: "PENDING", Value: "$(DYNAMO_RANK)"}},
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				Multinode: &v1beta1.MultinodeSpec{NodeCount: 2},
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main",
					Env: []corev1.EnvVar{{Name: "Z_MODE", Value: "pod"}, {Name: consts.DynamoRankEnvVar, Value: "$(PENDING)"}},
				}}}},
			}},
		},
		Status: v1beta1.DynamoGraphDeploymentStatus{Components: map[string]v1beta1.ComponentReplicaStatus{"worker": {RuntimeNamespace: "ns-graph-workers"}}},
	}
	original := dgd.DeepCopy()
	supported, err := supportsPodDiscovery(dgd, configv1alpha1.DiscoveryBackendKubernetes)
	require.NoError(t, err)
	require.True(t, supported)
	require.Equal(t, original, dgd)

	t.Log("Reconciliation publishes the relay instead of withdrawing a supported workload")
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{v1alpha1.AddToScheme, v1beta1.AddToScheme, appsv1.AddToScheme, corev1.AddToScheme, rbacv1.AddToScheme} {
		require.NoError(t, add(scheme))
	}
	relay := &v1alpha1.DynamoPoolRelay{Name: "graph", Namespace: "ns", UID: "relay-uid", Spec: relaySpec("cluster")}
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd, relay).WithStatusSubresource(dgd, relay).Build()
	reconciler := &reconciler{Client: kube, scheme: scheme, defaultBackend: configv1alpha1.DiscoveryBackendKubernetes}
	_, err = reconciler.Reconcile(t.Context(), ctrl.Request{Name: "graph", Namespace: "ns"})
	require.NoError(t, err)
	key := types.NamespacedName{Name: "graph-pool-relay", Namespace: "ns"}
	require.NoError(t, kube.Get(t.Context(), key, &corev1.Service{}))
	require.NoError(t, kube.Get(t.Context(), key, &appsv1.Deployment{}))
}
