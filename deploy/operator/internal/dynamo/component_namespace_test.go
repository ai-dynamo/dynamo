/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestComponentNamespacePrefixRuntimeCompatibility(t *testing.T) {
	t.Log("render frontend and EPP defaults across the strict-discovery release boundary")
	versions := []struct {
		name    string
		version *runtimeversion.Version
		strict  bool
	}{
		{name: "unknown runtime"},
		{name: "older runtime", version: &runtimeversion.Version{Major: 1, Minor: 5, Patch: 9}},
		{name: "supported runtime", version: &runtimeversion.Version{Major: 1, Minor: 6, Patch: 0}, strict: true},
		{name: "newer runtime", version: &runtimeversion.Version{Major: 1, Minor: 7, Patch: 0}, strict: true},
	}

	for _, componentType := range []string{commonconsts.ComponentTypeFrontend, commonconsts.ComponentTypeEPP} {
		for _, version := range versions {
			t.Run(componentType+"/"+version.name, func(t *testing.T) {
				t.Log("generate the component's production container defaults")
				container, err := ComponentDefaultsFactory(componentType).GetBaseContainer(ComponentContext{
					DynamoNamespace: "default-foo",
					ComponentType:   componentType,
					RuntimeVersion:  version.version,
				})
				require.NoError(t, err)

				t.Log("preserve the base namespace and change only supported runtime defaults")
				env := envVarsToMap(container.Env)
				assert.Equal(t, "default-foo", env[commonconsts.DynamoNamespacePrefixEnvVar])
				if version.strict {
					assert.Equal(t, "true", env[commonconsts.DynamoNamespacePrefixStrictEnvVar])
				} else {
					assert.NotContains(t, env, commonconsts.DynamoNamespacePrefixStrictEnvVar)
				}
			})
		}
	}
}

func TestLegacyEPPDoesNotEnableStrictNamespacePrefix(t *testing.T) {
	container, err := NewEPPDefaults().GetBaseContainer(ComponentContext{
		DynamoNamespace: "default-foo",
		ComponentType:   commonconsts.ComponentTypeEPP,
		RuntimeVersion:  &runtimeversion.Version{Major: 1, Minor: 6, Patch: 0},
		EPPConfig:       &v1beta1.EPPConfig{},
	})
	require.NoError(t, err)
	env := envVarsToMap(container.Env)
	assert.Equal(t, "default-foo", env[commonconsts.DynamoNamespacePrefixEnvVar])
	assert.NotContains(t, env, commonconsts.DynamoNamespacePrefixStrictEnvVar)
	assert.Contains(t, container.Args, "--pool-name")
}

func TestFrontendSidecarNamespacePrefixRuntimeCompatibility(t *testing.T) {
	tests := []struct {
		name            string
		workerImage     string
		sidecarImage    string
		workerOverride  string
		runtimeImage    string
		sidecarEnv      []corev1.EnvVar
		wantStrictValue string
	}{
		{name: "older worker and frontend", workerImage: "worker:1.5.0", sidecarImage: "frontend:1.5.0"},
		{name: "older worker with supported frontend", workerImage: "worker:1.5.0", sidecarImage: "frontend:1.6.0", wantStrictValue: "true"},
		{name: "supported worker with older frontend", workerImage: "worker:1.6.0", sidecarImage: "frontend:1.5.0"},
		{name: "supported worker and frontend", workerImage: "worker:1.6.0", sidecarImage: "frontend:1.6.0", wantStrictValue: "true"},
		{name: "worker override does not apply to unknown frontend", workerImage: "worker:custom", workerOverride: "1.6.0", sidecarImage: "frontend:custom"},
		{name: "digest-only frontend stays unknown", workerImage: "worker:1.6.0", sidecarImage: "frontend@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
		{name: "custom frontend explicitly enables strict mode", workerImage: "worker:1.5.0", sidecarImage: "frontend:custom", sidecarEnv: []corev1.EnvVar{{Name: commonconsts.DynamoNamespacePrefixStrictEnvVar, Value: "true"}}, wantStrictValue: "true"},
		{name: "supported frontend explicitly disables strict mode", workerImage: "worker:1.5.0", sidecarImage: "frontend:1.6.0", sidecarEnv: []corev1.EnvVar{{Name: commonconsts.DynamoNamespacePrefixStrictEnvVar, Value: "false"}}, wantStrictValue: "false"},
		{name: "older runtime init sidecar with supported frontend", workerImage: "engine:custom", runtimeImage: "runtime:1.5.0", sidecarImage: "frontend:1.6.0", wantStrictValue: "true"},
		{name: "supported runtime init sidecar with older frontend", workerImage: "engine:custom", runtimeImage: "runtime:1.6.0", sidecarImage: "frontend:1.5.0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("configure independently versioned containers with container discovery")
			sidecarName := "sidecar-frontend"
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType:          v1beta1.ComponentTypeWorker,
				RuntimeVersionOverride: tt.workerOverride,
				FrontendSidecar:        &sidecarName,
				PodTemplate: &corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: "container",
				}}, Spec: corev1.PodSpec{Containers: []corev1.Container{
					{Name: commonconsts.MainContainerName, Image: tt.workerImage},
					{Name: sidecarName, Image: tt.sidecarImage, Env: tt.sidecarEnv},
					{Name: "other-sidecar", Image: "frontend:1.7.0"},
				}}},
			}

			// Native runtime layout must not change the independently versioned frontend gate.
			if tt.runtimeImage != "" {
				component.PodTemplate.Spec.InitContainers = []corev1.Container{{
					Name: commonconsts.RuntimeContainerName, Image: tt.runtimeImage,
					RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
				}}
			}

			t.Log("render the production pod spec, including frontend sidecar defaults")
			pod, err := GenerateBasePodSpec(component, BackendFrameworkVLLM, &mockSecretsRetriever{}, "foo", "default", RoleMain, 1, &configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, "test-service", nil, staticContainerGPUCount(0))
			require.NoError(t, err)
			require.Len(t, pod.Containers, 3)

			t.Log("use the designated frontend image and preserve explicit environment overrides")
			sidecar := pod.Containers[1]
			assert.Equal(t, sidecarName, sidecar.Name)
			assert.Equal(t, tt.sidecarImage, sidecar.Image)
			env := envVarsToMap(sidecar.Env)
			assert.Equal(t, sidecarName, env["CONTAINER_NAME"])
			assert.Equal(t, "default-foo", env[commonconsts.DynamoNamespacePrefixEnvVar])
			if tt.wantStrictValue != "" {
				assert.Equal(t, tt.wantStrictValue, env[commonconsts.DynamoNamespacePrefixStrictEnvVar])
			} else {
				assert.NotContains(t, env, commonconsts.DynamoNamespacePrefixStrictEnvVar)
			}
			assert.NotContains(t, envVarsToMap(pod.Containers[2].Env), commonconsts.DynamoNamespacePrefixStrictEnvVar)
		})
	}
}
