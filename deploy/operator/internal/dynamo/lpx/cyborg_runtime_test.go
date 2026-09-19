/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestApplyCyborgManifestPathPrecedesAuthoredReferences(t *testing.T) {
	t.Parallel()

	t.Log("Define authored bindings that depend on the generated manifest location")
	projection := &ModelProjection{configuredBuild: Build{Path: "file:///models/build"}}
	authored := []corev1.EnvVar{
		{Name: "MODEL_PATH", Value: "$(GBUILD_MANIFEST_PATH)"},
		{Name: "OTHER", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
	}
	want := append([]corev1.EnvVar{{Name: gbuildManifestPathEnv, Value: "/models/build/manifest.v2.capnp.bin"}}, authored...)
	for _, test := range []struct {
		name string
		env  []corev1.EnvVar
	}{
		{name: "missing binding", env: authored},
		{name: "stale binding after reference", env: append(append([]corev1.EnvVar(nil), authored...), corev1.EnvVar{
			Name: gbuildManifestPathEnv, Value: "/stale/manifest",
		})},
		{name: "existing binding before reference", env: want},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Publish the authoritative path before references while preserving other bindings")
			container := corev1.Container{Env: test.env}
			require.NoError(t, applyCyborgManifestPath(&container, projection, "/models"))
			require.Equal(t, want, container.Env)

			t.Log("Repeat rendering without duplicating or reordering environment bindings")
			require.NoError(t, applyCyborgManifestPath(&container, projection, "/models"))
			require.Equal(t, want, container.Env)
		})
	}
}
