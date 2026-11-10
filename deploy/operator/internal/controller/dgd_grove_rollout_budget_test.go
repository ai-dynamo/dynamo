// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGroveRendererRolloutBudgetSurvivesTemplateSeed(t *testing.T) {
	for _, grouped := range []bool{false, true} {
		for _, tc := range []struct {
			name                  string
			seed, desired, budget int32
		}{
			{name: "scale up", seed: 1, desired: 4, budget: 2},
			{name: "scale below budget", seed: 4, desired: 2, budget: 3},
			{name: "scale to zero", seed: 1, desired: 0, budget: 2},
		} {
			t.Run(fmt.Sprintf("grouped=%t/%s", grouped, tc.name), func(t *testing.T) {
				t.Log("Create the existing PCS with its creation-time replica seed")
				dgd := &v1beta1.DynamoGraphDeployment{
					Name: "graph", Namespace: "default", UID: "dgd-uid",
					Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
						ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: new(tc.seed),
						PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "runtime:1.6.0"}}}},
					}}},
				}
				if grouped {
					dgd.Spec.Components[0].Experimental = &v1beta1.ExperimentalSpec{Grove: &v1beta1.GroveSpec{ForceScalingGroup: new(true)}}
				}
				config := &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}}
				runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}
				secrets := &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}
				existing, err := dynamo.GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, runtimeConfig, nil, secrets, nil, nil, false, nil)
				require.NoError(t, err)
				existing.UID = "pcs-uid"
				existing.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
				baseClient := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(existing).Build()
				reader := interceptor.NewClient(baseClient, interceptor.Funcs{
					Update: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.UpdateOption) error {
						// Model only Grove's admission response; this does not validate Grove compatibility.
						wire := object.(*unstructured.Unstructured)
						field := "cliques"
						if grouped {
							field = "podCliqueScalingGroups"
						}
						items, _, err := unstructured.NestedSlice(wire.Object, "spec", "template", field)
						require.NoError(t, err)
						for _, raw := range items {
							item := raw.(map[string]any)
							if _, exists := item["rollingUpdate"]; !exists {
								item["rollingUpdate"] = map[string]any{"maxUnavailable": int64(1)}
							}
						}
						require.NoError(t, unstructured.SetNestedSlice(wire.Object, items, "spec", "template", field))
						return delegated.Update(ctx, object, opts...)
					},
				})
				syncer := &groveWorkloadsReconciler{syncer: newDGDResourceSyncer(reader, events.NewFakeRecorder(20))}
				renderer := groveWorkloadRenderer{reader: reader, config: config, runtimeConfig: runtimeConfig, dockerSecretRetriever: secrets}

				t.Log("Render a new capacity and budget while preserving the existing seed")
				component := &dgd.Spec.Components[0]
				component.Replicas = new(tc.desired)
				component.ProviderOverride = &v1beta1.ProviderOverride{APIVersion: provideroverride.GroveAPIVersion, Value: apiextensionsv1.JSON{Raw: []byte(fmt.Sprintf(`{"rollingUpdate":{"maxUnavailable":%d}}`, tc.budget))}}
				provideroverride.DefaultTarget(component.ProviderOverride, "grove", provideroverride.ScopeComponent, component)
				original := dgd.DeepCopy()
				rendered, err := renderer.Render(t.Context(), groveReconcileRequest{DGD: dgd}, nil, nil, false)
				require.NoError(t, err)
				wire, err := provideroverride.ComposeGroveOverrides(dgd, rendered.desired)
				require.NoError(t, err)
				var pcs grovev1alpha1.PodCliqueSet
				require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(wire.Object, &pcs))

				t.Log("Verify the authored budget remains intact independently of the seed and desired replicas")
				if grouped {
					require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 1)
					group := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0]
					require.Equal(t, tc.seed, *group.Replicas)
					require.Equal(t, tc.budget, *group.RollingUpdate.MaxUnavailable)
					require.Nil(t, pcs.Spec.Template.Cliques[0].RollingUpdate)
				} else {
					require.Len(t, pcs.Spec.Template.Cliques, 1)
					require.Equal(t, tc.seed, pcs.Spec.Template.Cliques[0].Spec.Replicas)
					require.Equal(t, tc.budget, *pcs.Spec.Template.Cliques[0].RollingUpdate.MaxUnavailable)
				}
				require.Equal(t, original, dgd)

				t.Log("Synchronize the composed budget into the existing PCS")
				synced, written, err := syncer.reconcilePodCliqueSet(t.Context(), dgd, rendered)
				require.NoError(t, err)
				require.True(t, written)
				require.Equal(t, existing.UID, synced.UID)
				originalTemplates := synced.Spec.Template.DeepCopy()
				for _, clique := range originalTemplates.Cliques {
					clique.RollingUpdate = nil
				}
				for i := range originalTemplates.PodCliqueScalingGroupConfigs {
					originalTemplates.PodCliqueScalingGroupConfigs[i].RollingUpdate = nil
				}

				for _, step := range []struct {
					name, fragment string
					want           int32
				}{
					{name: "change the budget", fragment: fmt.Sprintf(`{"rollingUpdate":{"maxUnavailable":%d}}`, tc.budget+1), want: tc.budget + 1},
					{name: "remove the budget and retain the minimum", fragment: `{"spec":{"minAvailable":1}}`, want: 1},
				} {
					t.Log(step.name)
					if grouped && step.want == 1 {
						step.fragment = `{"minAvailable":1}`
					}
					component.ProviderOverride.Value.Raw = []byte(step.fragment)
					next, err := renderer.Render(t.Context(), groveReconcileRequest{DGD: dgd}, nil, nil, false)
					require.NoError(t, err)
					synced, written, err = syncer.reconcilePodCliqueSet(t.Context(), dgd, next)
					require.NoError(t, err)
					require.True(t, written)
					require.Equal(t, existing.UID, synced.UID)
					if grouped {
						require.Equal(t, step.want, *synced.Spec.Template.PodCliqueScalingGroupConfigs[0].RollingUpdate.MaxUnavailable)
					} else {
						require.Equal(t, step.want, *synced.Spec.Template.Cliques[0].RollingUpdate.MaxUnavailable)
					}
					templates := synced.Spec.Template.DeepCopy()
					for _, clique := range templates.Cliques {
						clique.RollingUpdate = nil
					}
					for i := range templates.PodCliqueScalingGroupConfigs {
						templates.PodCliqueScalingGroupConfigs[i].RollingUpdate = nil
					}
					require.Equal(t, originalTemplates, templates)
				}

				t.Log("Retain Grove's defaulted policy on repeated reconciliation after removal")
				next, err := renderer.Render(t.Context(), groveReconcileRequest{DGD: dgd}, nil, nil, false)
				require.NoError(t, err)
				_, written, err = syncer.reconcilePodCliqueSet(t.Context(), dgd, next)
				require.NoError(t, err)
				require.False(t, written)

			})
		}
	}
}
