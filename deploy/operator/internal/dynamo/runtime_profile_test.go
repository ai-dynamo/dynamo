/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func TestRuntimeProfileForComponent(t *testing.T) {
	tests := []struct {
		name     string
		image    string
		override string
		want     bool
	}{
		{
			name:  "resolves the image tag",
			image: "nvcr.io/nvidia/ai-dynamo/runtime:v1.5.0-cuda13",
			want:  true,
		},
		{
			name:     "resolves the override",
			image:    "registry.example/runtime:custom",
			override: "1.5.0",
			want:     true,
		},
		{
			name:     "the override is authoritative",
			image:    "nvcr.io/nvidia/ai-dynamo/runtime:1.5.0",
			override: "1.4.0",
		},
		{
			name:  "runtime 1.4 keeps the legacy profile",
			image: "nvcr.io/nvidia/ai-dynamo/runtime:1.4.0",
		},
		{
			name:  "an unresolvable legacy image keeps the legacy profile",
			image: "registry.example/runtime:custom",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName:          "worker",
				ComponentType:          commonconsts.ComponentTypeWorker,
				RuntimeVersionOverride: tt.override,
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  commonconsts.MainContainerName,
							Image: tt.image,
						}},
					},
				},
			}

			got := runtimeProfileForComponent(component)
			if got.CanaryHealthChecks != tt.want {
				t.Fatalf("CanaryHealthChecks = %t, want %t", got.CanaryHealthChecks, tt.want)
			}
		})
	}
}
