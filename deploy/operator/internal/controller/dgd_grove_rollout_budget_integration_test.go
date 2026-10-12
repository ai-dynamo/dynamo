//go:build groveintegration

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"os"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
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
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGroveRolloutBudgetsAgainstInstalledWebhook(t *testing.T) {
	t.Log("Connect only to the dedicated Kind cluster installed by make grove-admission")
	kubeconfig := os.Getenv("DYNAMO_GROVE_ADMISSION_KUBECONFIG")
	contextName := os.Getenv("DYNAMO_GROVE_ADMISSION_CONTEXT")
	require.NotEmpty(t, kubeconfig)
	require.NotEmpty(t, contextName)
	config, err := clientcmd.LoadFromFile(kubeconfig)
	require.NoError(t, err)
	require.Equal(t, contextName, config.CurrentContext)
	restConfig, err := clientcmd.NewDefaultClientConfig(*config, &clientcmd.ConfigOverrides{CurrentContext: contextName}).ClientConfig()
	require.NoError(t, err)
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	apiClient, err := client.New(restConfig, client.Options{Scheme: scheme})
	require.NoError(t, err)

	for _, grouped := range []bool{false, true} {
		for _, tc := range []struct {
			name                           string
			seed, desired, minimum, budget int32
			strategy                       grovev1alpha1.UpdateStrategyType
			omitBudget                     bool
			wantBudget                     int32
			wantError                      string
		}{
			{name: "scaled above seed", seed: 1, desired: 4, minimum: 1, budget: 2, wantBudget: 2},
			{name: "scaled below seed", seed: 4, desired: 2, minimum: 1, budget: 3, wantBudget: 3},
			{name: "scaled to zero", seed: 1, desired: 0, minimum: 1, budget: 2, wantBudget: 2},
			{name: "Coherent budget above minimum seed", seed: 2, desired: 4, minimum: 2, budget: 3, strategy: grovev1alpha1.CoherentStrategy, wantBudget: 3},
			{name: "RollingRecreate default", seed: 2, desired: 2, minimum: 2, omitBudget: true, wantBudget: 1},
			{name: "Coherent default", seed: 2, desired: 2, minimum: 2, strategy: grovev1alpha1.CoherentStrategy, omitBudget: true, wantBudget: 2},
			{name: "Coherent budget below minimum", seed: 2, desired: 2, minimum: 2, budget: 1, strategy: grovev1alpha1.CoherentStrategy, wantError: "must not be less than minAvailable"},
			{name: "OnDelete rejects rollout configuration", seed: 2, desired: 2, minimum: 1, budget: 1, strategy: grovev1alpha1.OnDeleteStrategy, wantError: "rollingUpdate must not be set"},
		} {
			t.Run(fmt.Sprintf("grouped=%t/%s", grouped, tc.name), func(t *testing.T) {
				t.Log("Render the creation-time PCS template and a later DGD capacity request")
				dgd := &v1beta1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{Name: "budget-compatibility", Namespace: "default", UID: "dgd-uid", Annotations: map[string]string{consts.KubeAnnotationGroveUpdateStrategy: string(tc.strategy)}},
					Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
						ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: ptr.To(tc.seed), MinAvailable: ptr.To(tc.minimum),
						PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example.invalid/frontend:test"}}}},
					}}},
				}
				if tc.strategy == "" {
					delete(dgd.Annotations, consts.KubeAnnotationGroveUpdateStrategy)
				}
				if grouped {
					dgd.Spec.Components[0].Experimental = &v1beta1.ExperimentalSpec{Grove: &v1beta1.GroveSpec{ForceScalingGroup: ptr.To(true)}}
				}
				operatorConfig := &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}}
				runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}
				secrets := &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}
				existing, err := dynamo.GenerateGrovePodCliqueSet(t.Context(), dgd, nil, operatorConfig, runtimeConfig, nil, secrets, nil, nil, false, nil)
				require.NoError(t, err)
				reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(existing).Build()
				renderer := groveWorkloadRenderer{reader: reader, config: operatorConfig, runtimeConfig: runtimeConfig, dockerSecretRetriever: secrets}
				component := &dgd.Spec.Components[0]
				component.Replicas = ptr.To(tc.desired)
				if !tc.omitBudget {
					component.ProviderOverride = &v1beta1.ProviderOverride{APIVersion: provideroverride.GroveAPIVersion, Value: apiextensionsv1.JSON{Raw: []byte(fmt.Sprintf(`{"rollingUpdate":{"maxUnavailable":%d}}`, tc.budget))}}
					provideroverride.DefaultTarget(component.ProviderOverride, consts.WorkloadProviderGrove, provideroverride.ScopeComponent, component)
				}
				rendered, err := renderer.Render(t.Context(), groveReconcileRequest{DGD: dgd}, nil, nil, false)
				require.NoError(t, err)
				wire, err := provideroverride.ComposeGroveOverrides(dgd, rendered.desired)
				require.NoError(t, err)

				t.Log("Run server-side dry-run through the installed Grove defaulting and validation webhooks")
				err = apiClient.Create(t.Context(), wire, client.DryRunAll)
				if tc.wantError != "" {
					require.ErrorContains(t, err, tc.wantError, "transparent webhook handlers must not satisfy this integration test")
					return
				}
				require.NoError(t, err)
				var admitted grovev1alpha1.PodCliqueSet
				require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(wire.Object, &admitted))
				if grouped {
					group := admitted.Spec.Template.PodCliqueScalingGroupConfigs[0]
					require.Equal(t, tc.seed, *group.Replicas)
					require.Equal(t, tc.wantBudget, *group.RollingUpdate.MaxUnavailable)
				} else {
					clique := admitted.Spec.Template.Cliques[0]
					require.Equal(t, tc.seed, clique.Spec.Replicas)
					require.Equal(t, tc.wantBudget, *clique.RollingUpdate.MaxUnavailable)
				}
			})
		}
	}
}
