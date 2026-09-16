// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secrets"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGenerateBasePodSpec_InitContainerPullSecrets(t *testing.T) {
	cases := []struct {
		name      string
		mainImage string
		explicit  []corev1.LocalObjectReference
		disabled  bool
		expected  []corev1.LocalObjectReference
	}{
		{name: "private init image", mainImage: "public.example/frontend:v1", expected: []corev1.LocalObjectReference{{Name: "init-pull-secret"}}},
		{name: "shared registry is deduplicated", mainImage: "init.example/frontend:v1", expected: []corev1.LocalObjectReference{{Name: "init-pull-secret"}}},
		{name: "explicit credentials preserved", mainImage: "public.example/frontend:v1", explicit: []corev1.LocalObjectReference{{Name: "explicit-secret"}}, expected: []corev1.LocalObjectReference{{Name: "explicit-secret"}, {Name: "init-pull-secret"}}},
		{name: "explicit credential is deduplicated", mainImage: "public.example/frontend:v1", explicit: []corev1.LocalObjectReference{{Name: "init-pull-secret"}}, expected: []corev1.LocalObjectReference{{Name: "init-pull-secret"}}},
		{name: "disabled discovery retains explicit credentials", mainImage: "public.example/frontend:v1", explicit: []corev1.LocalObjectReference{{Name: "explicit-secret"}}, disabled: true, expected: []corev1.LocalObjectReference{{Name: "explicit-secret"}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Index a credential for the private initialization image")
			credential := &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "init-pull-secret", Namespace: "default"},
				Type:       corev1.SecretTypeDockerConfigJson,
				Data:       map[string][]byte{corev1.DockerConfigJsonKey: []byte(`{"auths":{"init.example":{}}}`)},
			}
			reader := fake.NewClientBuilder().WithScheme(scheme.Scheme).WithObjects(credential).Build()
			index := secrets.NewDockerSecretIndexer(reader, "default")
			require.NoError(t, index.RefreshIndex(t.Context()))

			t.Log("Render a component with a private init container and the requested discovery policy")
			component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				ExtraPodSpec: &v1alpha1.ExtraPodSpec{
					MainContainer: &corev1.Container{Image: tc.mainImage},
					PodSpec: &corev1.PodSpec{
						InitContainers:   []corev1.Container{{Name: "prepare", Image: "init.example/prepare:v1"}},
						ImagePullSecrets: tc.explicit,
					},
				},
			}
			if tc.disabled {
				component.Annotations = map[string]string{commonconsts.KubeAnnotationDisableImagePullSecretDiscovery: commonconsts.KubeLabelValueTrue}
			}
			renderedComponent := betaComponent(t, component)
			original := renderedComponent.DeepCopy()
			pod, err := GenerateBasePodSpec(
				renderedComponent, BackendFrameworkNoop, index,
				"test-deployment", "default", RoleMain, 1,
				&configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove,
				"test-service", nil, staticContainerGPUCount(0),
			)
			require.NoError(t, err)

			t.Log("Verify discovery, stable ordering, deduplication, and preservation of the input component")
			require.Len(t, pod.InitContainers, 1)
			require.Equal(t, "init.example/prepare:v1", pod.InitContainers[0].Image)
			require.Equal(t, tc.expected, pod.ImagePullSecrets)
			require.Equal(t, original, renderedComponent)
		})
	}
}
