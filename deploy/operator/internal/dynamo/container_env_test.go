/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestSGLangBackend_EnvFromResolution(t *testing.T) {
	const namespace = "test"

	t.Log("define envFrom content that is unrelated, selects embedding mode, or overrides the payload")
	tests := []struct {
		name                string
		data                map[string]string
		secret              bool
		args                []string
		wantExplicitPayload bool
	}{
		{
			name:                "unrelated source does not block compatibility payload",
			data:                map[string]string{"UNRELATED": "value"},
			args:                []string{sglangEmbeddingWorkerFlag},
			wantExplicitPayload: true,
		},
		{
			name:                "embedding mode is resolved from source",
			data:                map[string]string{sglangEmbeddingWorkerEnv: booleanTrueValue},
			wantExplicitPayload: true,
		},
		{
			name:   "payload override remains owned by source",
			data:   map[string]string{healthCheckPayloadEnv: `{"model":"custom","input":"probe"}`},
			secret: true,
			args:   []string{sglangEmbeddingWorkerFlag},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("create the referenced ConfigMap or Secret")
			const sourceName = "worker-config"
			builder := fake.NewClientBuilder().WithScheme(scheme.Scheme)
			source := corev1.EnvFromSource{}
			if tt.secret {
				data := make(map[string][]byte, len(tt.data))
				for key, value := range tt.data {
					data[key] = []byte(value)
				}
				builder = builder.WithObjects(&corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{Name: sourceName, Namespace: namespace},
					Data:       data,
				})
				source.SecretRef = &corev1.SecretEnvSource{
					LocalObjectReference: corev1.LocalObjectReference{Name: sourceName},
				}
			} else {
				builder = builder.WithObjects(&corev1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: sourceName, Namespace: namespace},
					Data:       tt.data,
				})
				source.ConfigMapRef = &corev1.ConfigMapEnvSource{
					LocalObjectReference: corev1.LocalObjectReference{Name: sourceName},
				}
			}
			reader := builder.Build()

			t.Log("materialize only relevant values from the referenced source")
			container := &corev1.Container{
				Image:   "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.0",
				Args:    tt.args,
				EnvFrom: []corev1.EnvFromSource{source},
			}
			originalEnvFrom := append([]corev1.EnvFromSource(nil), container.EnvFrom...)
			resolved, overlay, err := MaterializeContainerEnvFrom(
				context.Background(),
				reader,
				namespace,
				container,
				sglangEmbeddingWorkerEnv,
				healthCheckPayloadEnv,
			)
			require.NoError(t, err)
			require.Empty(t, resolved.EnvFrom)

			t.Log("render the compatibility setting and restore the original envFrom source")
			backend := &SGLangBackend{}
			err = backend.UpdateContainer(
				resolved,
				1,
				RoleMain,
				&v1beta1.DynamoComponentDeploymentSharedSpec{},
				"test-service",
				&GroveMultinodeDeployer{},
				staticContainerGPUCount(0),
			)
			require.NoError(t, err)
			overlay.Restore(resolved)

			t.Log("verify unrelated sources allow injection while sourced values remain authoritative")
			require.Equal(t, originalEnvFrom, resolved.EnvFrom)
			payload := findEnvVar(resolved.Env, healthCheckPayloadEnv)
			if tt.wantExplicitPayload {
				require.NotNil(t, payload)
				require.Equal(t, sglang15EmbeddingHealthCheckPayload, payload.Value)
				return
			}
			require.Nil(t, payload)
		})
	}
}
