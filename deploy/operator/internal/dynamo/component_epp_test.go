/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func eppContainerFor(t *testing.T, eppConfig *nvidiacomv1beta1.EPPConfig) corev1.Container {
	t.Helper()
	container, err := NewEPPDefaults().GetBaseContainer(ComponentContext{
		DynamoNamespace:                "ns-dgd",
		ComponentType:                  commonconsts.ComponentTypeEPP,
		ParentGraphDeploymentName:      "dgd",
		ParentGraphDeploymentNamespace: "ns",
		EPPConfig:                      eppConfig,
	})
	require.NoError(t, err)
	return container
}

func mountPathNamed(container corev1.Container, name string) string {
	for _, m := range container.VolumeMounts {
		if m.Name == name {
			return m.MountPath
		}
	}
	return ""
}

func envValueNamed(container corev1.Container, name string) (string, bool) {
	for _, e := range container.Env {
		if e.Name == name {
			return e.Value, true
		}
	}
	return "", false
}

// The Rust EPP resolves its model-config cache root from $HOME
// (lib/llm/src/model_card.rs), so the hf-cache emptyDir has to be mounted under
// the HOME the container actually runs with. A mismatch is silently inert: the
// blobs go to the ephemeral writable layer instead of the volume.
func TestEPPCacheMountTracksHome(t *testing.T) {
	t.Run("native Rust EPP pins HOME and mounts under it", func(t *testing.T) {
		container := eppContainerFor(t, nil)

		home, ok := envValueNamed(container, "HOME")
		require.True(t, ok, "native EPP must pin HOME so the cache mount cannot drift per image")
		assert.Equal(t, nativeRustEPPHome, home)
		assert.Equal(t, home+"/.cache", mountPathNamed(container, "hf-cache"))
	})

	t.Run("legacy Go EPP keeps the nonroot cache path", func(t *testing.T) {
		container := eppContainerFor(t, &nvidiacomv1beta1.EPPConfig{})

		assert.Equal(t, legacyGoEPPHome+"/.cache", mountPathNamed(container, "hf-cache"),
			"legacy Go EPP images run as nonroot; moving this path would break existing DGDs")
	})
}

// The volume the mount refers to must exist, or the pod is rejected outright.
func TestEPPCacheVolumeIsDeclaredForBothContracts(t *testing.T) {
	for name, eppConfig := range map[string]*nvidiacomv1beta1.EPPConfig{
		"native": nil,
		"legacy": {},
	} {
		t.Run(name, func(t *testing.T) {
			podSpec, err := NewEPPDefaults().GetBasePodSpec(ComponentContext{
				ComponentType:                  commonconsts.ComponentTypeEPP,
				ParentGraphDeploymentName:      "dgd",
				ParentGraphDeploymentNamespace: "ns",
				EPPConfig:                      eppConfig,
			})
			require.NoError(t, err)

			found := false
			for _, v := range podSpec.Volumes {
				if v.Name == "hf-cache" {
					found = true
					assert.NotNil(t, v.EmptyDir)
				}
			}
			assert.True(t, found, "hf-cache volume must back the mount")
		})
	}
}
