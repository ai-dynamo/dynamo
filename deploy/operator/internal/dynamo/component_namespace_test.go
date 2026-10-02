/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
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

func TestFrontendSidecarNamespacePrefixRuntimeCompatibility(t *testing.T) {
	for _, minor := range []uint64{5, 6} {
		t.Run((&runtimeversion.Version{Major: 1, Minor: minor}).String(), func(t *testing.T) {
			pod := corev1.PodSpec{Containers: []corev1.Container{{Name: "sidecar-frontend"}}}
			err := mergeFrontendSidecarDefaults(&pod, "sidecar-frontend", ComponentContext{
				DynamoNamespace: "default-foo",
				RuntimeVersion:  &runtimeversion.Version{Major: 1, Minor: minor},
			}, &configv1alpha1.OperatorConfiguration{}, nil)
			require.NoError(t, err)
			env := envVarsToMap(pod.Containers[0].Env)
			assert.Equal(t, "default-foo", env[commonconsts.DynamoNamespacePrefixEnvVar])
			if minor >= 6 {
				assert.Equal(t, "true", env[commonconsts.DynamoNamespacePrefixStrictEnvVar])
			} else {
				assert.NotContains(t, env, commonconsts.DynamoNamespacePrefixStrictEnvVar)
			}
		})
	}
}
