// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
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
					ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: "dgd-uid"},
					Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
						ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: ptr.To(tc.seed),
						PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "runtime:1.6.0"}}}},
					}}},
				}
				if grouped {
					dgd.Spec.Components[0].Experimental = &v1beta1.ExperimentalSpec{Grove: &v1beta1.GroveSpec{ForceScalingGroup: ptr.To(true)}}
				}
				config := &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}}
				runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}
				secrets := &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}
				existing, err := dynamo.GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, runtimeConfig, nil, secrets, nil, nil, false, nil)
				require.NoError(t, err)
				reader := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(existing).Build()
				renderer := groveWorkloadRenderer{reader: reader, config: config, runtimeConfig: runtimeConfig, dockerSecretRetriever: secrets}

				t.Log("Render a new capacity and budget while preserving the existing seed")
				component := &dgd.Spec.Components[0]
				component.Replicas = ptr.To(tc.desired)
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
			})
		}
	}
}
