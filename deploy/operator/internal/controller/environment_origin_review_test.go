// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDCDEnvironmentOriginUsesOnlyGraphController(t *testing.T) {
	for _, test := range []struct {
		name, objectOrigin, templateOrigin, ownerVersion, ownerKind string
		controller, ordered                                         bool
	}{
		{name: "non-controlling DGD keeps standalone semantic origin", objectOrigin: "1.8.0", templateOrigin: "1.7.0", ownerVersion: "nvidia.com/v1beta1", ownerKind: "DynamoGraphDeployment", ordered: true},
		{name: "foreign DGD controller keeps standalone semantic origin", objectOrigin: "1.8.0", templateOrigin: "1.7.0", ownerVersion: "example.com/v1", ownerKind: "DynamoGraphDeployment", controller: true, ordered: true},
		{name: "unrelated controller keeps standalone semantic origin", objectOrigin: "1.8.0", templateOrigin: "1.7.0", ownerVersion: "nvidia.com/v1beta1", ownerKind: "OtherResource", controller: true, ordered: true},
		{name: "non-controlling DGD cannot enable template semantics", objectOrigin: "1.7.0", templateOrigin: "1.8.0", ownerVersion: "nvidia.com/v1beta1", ownerKind: "DynamoGraphDeployment"},
		{name: "foreign DGD cannot enable template semantics", objectOrigin: "1.7.0", templateOrigin: "1.8.0", ownerVersion: "example.com/v1", ownerKind: "DynamoGraphDeployment", controller: true},
		{name: "beta DGD controller selects inherited origin", objectOrigin: "1.8.0", templateOrigin: "1.7.0", ownerVersion: "nvidia.com/v1beta1", ownerKind: "DynamoGraphDeployment", controller: true},
		{name: "alpha DGD controller selects inherited origin", objectOrigin: "1.7.0", templateOrigin: "1.8.0", ownerVersion: "nvidia.com/v1alpha1", ownerKind: "DynamoGraphDeployment", controller: true, ordered: true},
		{name: "standalone missing origin cannot enable template semantics", templateOrigin: "1.8.0"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Use only the controlling nvidia.com DGD reference for the inherited origin")
			dcd := &v1beta1.DynamoComponentDeployment{Name: "frontend", Namespace: "ns",
				Annotations:     map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: test.objectOrigin},
				OwnerReferences: []metav1.OwnerReference{{APIVersion: test.ownerVersion, Kind: test.ownerKind, Name: "graph", UID: "graph-uid", Controller: new(test.controller)}},
				Spec: v1beta1.DynamoComponentDeploymentSpec{BackendFramework: "vllm", DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
					PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: test.templateOrigin}, Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "runtime:1.7.0", Env: []corev1.EnvVar{{Name: "B", Value: "$(A)"}, {Name: "A", Value: "x"}}}},
					}},
				}},
			}
			if test.ownerKind == "" {
				dcd.OwnerReferences = nil
			}
			require.Equal(t, test.ordered, dynamo.DCDUsesOrderedEnvironment(dcd))
			kubeClient := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(dcd).Build()
			original := dcd.DeepCopy()
			reconciler := &DynamoComponentDeploymentReconciler{Client: kubeClient, Recorder: events.NewFakeRecorder(10), Config: &configv1alpha1.OperatorConfiguration{}, RuntimeConfig: &common.RuntimeConfig{}}
			deployment, _, err := reconciler.generateDeployment(t.Context(), generateResourceOption{dynamoComponentDeployment: dcd})
			require.NoError(t, err)
			requireEnvironmentOrder(t, deployment.Spec.Template.Spec.Containers[0].Env, test.ordered)
			require.Equal(t, original, dcd, "origin selection must not modify the input")
		})
	}
}

func requireEnvironmentOrder(t *testing.T, env []corev1.EnvVar, forward bool) {
	t.Helper()
	var actual []corev1.EnvVar
	for _, variable := range env {
		if variable.Name == "A" || variable.Name == "B" {
			actual = append(actual, variable)
		}
	}
	want := []corev1.EnvVar{{Name: "A", Value: "x"}, {Name: "B", Value: "$(A)"}}
	if forward {
		want[0], want[1] = want[1], want[0]
	}
	require.Equal(t, want, actual)
}

func TestDCDEnvironmentOriginCanDifferBetweenRoles(t *testing.T) {
	t.Log("A graph child inherits each role's origin, and any semantic role makes LWS hashing order-sensitive")
	leader := &corev1.PodTemplateSpec{Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"}, Spec: corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Image: "runtime:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"}, Args: []string{"--model", "test/model"},
			Env: []corev1.EnvVar{{Name: "B", Value: "$(A)"}, {Name: "A", Value: "x"}}}},
	}}
	worker := leader.DeepCopy()
	worker.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = "1.8.0"
	dcd := &v1beta1.DynamoComponentDeployment{Name: "worker", Namespace: "ns",
		OwnerReferences: []metav1.OwnerReference{{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment", Name: "graph", Controller: new(true)}},
		Spec: v1beta1.DynamoComponentDeploymentSpec{BackendFramework: "vllm", DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker, Multinode: &v1beta1.MultinodeSpec{NodeCount: 2},
			Roles: []v1beta1.ComponentRoleSpec{{Name: "leader", PodTemplate: leader}, {Name: "worker", PodTemplate: worker}},
		}},
	}
	require.True(t, dynamo.DCDUsesOrderedEnvironment(dcd))
	kubeClient := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(dcd).Build()
	reconciler := &DynamoComponentDeploymentReconciler{Client: kubeClient, Recorder: events.NewFakeRecorder(10), Config: &configv1alpha1.OperatorConfiguration{}, RuntimeConfig: &common.RuntimeConfig{}}
	workload, _, err := reconciler.generateLeaderWorkerSet(t.Context(), generateResourceOption{dynamoComponentDeployment: dcd}, dcd.Name)
	require.NoError(t, err)
	requireEnvironmentOrder(t, workload.Spec.LeaderWorkerTemplate.LeaderTemplate.Spec.Containers[0].Env, false)
	requireEnvironmentOrder(t, workload.Spec.LeaderWorkerTemplate.WorkerTemplate.Spec.Containers[0].Env, true)
}
