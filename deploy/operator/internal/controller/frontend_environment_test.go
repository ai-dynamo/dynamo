// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"slices"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestStandaloneFrontendEnvironmentUsesDCDOrigin(t *testing.T) {
	for _, origin := range []string{"1.8.0", "1.7.0", ""} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("The standalone DCD origin overrides a conflicting template origin")
			podOrigin := "1.8.0"
			if origin == "1.8.0" {
				podOrigin = "1.7.0"
			}
			dcd := &v1beta1.DynamoComponentDeployment{Name: "gateway", Namespace: "ns",
				Spec: v1beta1.DynamoComponentDeploymentSpec{
					Env: []corev1.EnvVar{{Name: consts.DynamoDeploymentConfigEnvVar, Value: `{"gateway":{"port":1234}}`}},
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentType: v1beta1.ComponentTypeFrontend,
						PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: podOrigin},
							Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend:1.7.0", Env: []corev1.EnvVar{{Name: "CFG", Value: "$(DYN_DEPLOYMENT_CONFIG)"}}}}},
						},
					},
				},
			}
			if origin != "" {
				dcd.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: origin}
			}
			original := dcd.DeepCopy()

			t.Log("Render through the production standalone adapter, including origin propagation")
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			renderer := newDCDWorkloadRenderer(fake.NewClientBuilder().WithScheme(scheme).Build(),
				&configv1alpha1.OperatorConfiguration{}, &commoncontroller.RuntimeConfig{}, nil)
			pod, err := renderer.generatePodTemplateSpec(t.Context(), dcd, dynamo.RoleMain, noContainerGPUs())
			require.NoError(t, err)
			var actual []corev1.EnvVar
			for _, variable := range pod.Spec.Containers[0].Env {
				if variable.Name == "CFG" || variable.Name == consts.DynamoDeploymentConfigEnvVar {
					actual = append(actual, variable)
				}
			}
			want := []corev1.EnvVar{{Name: "CFG", Value: "$(DYN_DEPLOYMENT_CONFIG)"}, {Name: consts.DynamoDeploymentConfigEnvVar, Value: `{"gateway":{"port":1234}}`}}
			if origin == "1.8.0" {
				want[1].Value = `{"gateway":{"port":8000}}`
				slices.Reverse(want)
			}
			require.Equal(t, want, actual)
			require.Equal(t, original, dcd)
		})
	}
}
