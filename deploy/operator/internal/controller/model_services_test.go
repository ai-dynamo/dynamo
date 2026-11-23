// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/disaggregatedset"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/modelservice"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func TestDCDModelServicePolicyUsesInheritedParentOrigin(t *testing.T) {
	for _, tc := range []struct {
		name, parentOrigin, childOrigin string
		managed                         bool
		want                            bool
		roleOnly                        bool
	}{
		{name: "new graph", parentOrigin: "1.8.0", childOrigin: "1.8.0", managed: true, want: true},
		{name: "new child of old graph", parentOrigin: "1.7.0", childOrigin: "1.8.0", managed: true},
		{name: "missing graph origin", childOrigin: "1.8.0", managed: true},
		{name: "invalid graph origin", parentOrigin: "invalid", childOrigin: "1.8.0", managed: true},
		{name: "old child metadata", parentOrigin: "1.8.0", childOrigin: "1.7.0", managed: true, want: true},
		{name: "standalone", parentOrigin: "1.8.0", childOrigin: "1.8.0"},
		{name: "complete role templates", parentOrigin: "1.8.0", childOrigin: "1.8.0", managed: true, want: true, roleOnly: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Classify model discovery from graph origin rather than child creation version")
			dcd := &v1beta1.DynamoComponentDeployment{
				Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: tc.childOrigin},
				Spec: v1beta1.DynamoComponentDeploymentSpec{DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
					PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: tc.parentOrigin}},
				}},
			}
			if tc.roleOnly {
				dcd.Spec.Roles = []v1beta1.ComponentRoleSpec{
					{Name: v1beta1.ComponentRoleLeader, PodTemplate: dcd.Spec.PodTemplate.DeepCopy()},
					{Name: v1beta1.ComponentRoleWorker, PodTemplate: dcd.Spec.PodTemplate.DeepCopy()},
				}
				dcd.Spec.PodTemplate = nil
			}
			if tc.managed {
				dcd.OwnerReferences = []metav1.OwnerReference{{APIVersion: v1beta1.GroupVersion.String(), Kind: dynamoGraphDeploymentKind, Name: "parent", UID: "parent-uid", Controller: new(true)}}
			}
			require.Equal(t, tc.want, dcdHasGraphOwnedModelServices(dcd))
		})
	}
}

func TestModelDiscoveryServiceProviderParity(t *testing.T) {
	for _, provider := range []string{"component", "grove", "disaggregatedset"} {
		for _, origin := range []string{"1.7.0", "1.8.0"} {
			t.Run(provider+"/"+origin, func(t *testing.T) {
				t.Log("Reconcile the provider with an explicit immutable graph origin")
				dgd := &v1beta1.DynamoGraphDeployment{
					Name: "graph", Namespace: "default", UID: "graph-uid",
					Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: origin},
					Spec: v1beta1.DynamoGraphDeploymentSpec{
						BackendFramework: "vllm",
						Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
							ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
							ModelRef: &v1beta1.ModelReference{Name: "llama-3"},
							PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
								Name: "main", Image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.7.0",
								Command: []string{"python3", "-m", "dynamo.vllm"}, Args: []string{"--model", "llama-3"},
							}}}},
						}},
					},
				}
				r := createTestDGDReconcilerWithStatus(dgd)
				r.Config.RBAC.PlannerClusterRoleName = "planner"
				r.Config.RBAC.EPPClusterRoleName = "epp"
				require.NoError(t, rbacv1.AddToScheme(r.Scheme()))
				require.NoError(t, appsv1.AddToScheme(r.Scheme()))
				require.NoError(t, r.Create(t.Context(), &rbacv1.ClusterRole{Name: "planner"}))
				require.NoError(t, r.Create(t.Context(), &rbacv1.ClusterRole{Name: "epp"}))
				switch provider {
				case "component":
					_, err := r.newComponentProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
					require.NoError(t, err)
				case "grove":
					r.RuntimeConfig.Gate = features.Gates{Grove: true}
					_, err := r.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
					require.NoError(t, err)
				case "disaggregatedset":
					renderer := newDCDWorkloadRenderer(r.Client, r.Config, r.RuntimeConfig, nil)
					stable := newDisaggregatedSetStableResourcesReconciler(r.Client, renderer)
					_, err := stable.Reconcile(t.Context(), dgd, nil, dynamo.ComponentsByName(dgd), disaggregatedset.Selection{}, "", false, dynamo.RollingUpdateContext{})
					require.NoError(t, err)
				}

				t.Log("New graphs use the same graph-owned model Service independently of provider")
				name := modelservice.Name("llama-3", dgd.Name, true)
				service := &corev1.Service{}
				err := r.Get(t.Context(), client.ObjectKey{Namespace: dgd.Namespace, Name: name}, service)
				if origin == "1.8.0" {
					require.NoError(t, err)
					require.True(t, metav1.IsControlledBy(service, dgd))
					require.Equal(t, dgd.Name, service.Spec.Selector[consts.KubeLabelDynamoGraphDeploymentName])
				} else {
					require.True(t, apierrors.IsNotFound(err), "%v", err)
					if provider != "component" {
						graph := ""
						if provider == "disaggregatedset" {
							graph = dgd.Name
						}
						require.NoError(t, r.Get(t.Context(), client.ObjectKey{Namespace: dgd.Namespace, Name: modelservice.Name("llama-3", graph, false)}, service))
					}
				}
			})
		}
	}
}

func TestModelServiceNamesCannotOverwriteComponentServices(t *testing.T) {
	for _, existing := range []bool{false, true} {
		t.Run(map[bool]string{false: "new graph", true: "retiring model"}[existing], func(t *testing.T) {
			t.Log("A component name must not claim the model-discovery Service's name")
			dgd := &v1beta1.DynamoGraphDeployment{
				Name: "graph", Namespace: "default", UID: "graph-uid",
				Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
					{ComponentName: "worker", ModelRef: &v1beta1.ModelReference{Name: "llama-3"}},
					{ComponentName: "model-e3352e1e", ComponentType: v1beta1.ComponentTypeFrontend},
				}},
			}
			kubeClient := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(dgd).Build()
			if existing {
				service := modelservice.Render(modelservice.Params{Namespace: dgd.Namespace, ModelName: "llama-3", GraphName: dgd.Name, ReadableName: true})
				service.Labels[modelservice.LabelModelDiscovery] = "true"
				require.NoError(t, controllerutil.SetControllerReference(dgd, service, kubeClient.Scheme()))
				require.NoError(t, kubeClient.Create(t.Context(), service))
				dgd.Spec.Components[0].ModelRef = nil
			}
			reconciler := modelservice.Reconciler{Client: kubeClient}
			_, err := reconciler.Reconcile(t.Context(), dgd)
			require.ErrorContains(t, err, "conflicts")
			services := &corev1.ServiceList{}
			require.NoError(t, kubeClient.List(t.Context(), services))
			if existing {
				require.Len(t, services.Items, 1)
				require.Equal(t, corev1.ClusterIPNone, services.Items[0].Spec.ClusterIP)
				require.Equal(t, int32(9090), services.Items[0].Spec.Ports[0].Port)
			} else {
				require.Empty(t, services.Items)
			}
		})
	}
}
