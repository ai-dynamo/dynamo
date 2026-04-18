// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package common

import (
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func TestResolveMainContainer(t *testing.T) {
	for _, tc := range []struct {
		name     string
		spec     *corev1.PodSpec
		wantName string
		wantNil  bool
	}{
		{
			name:    "nil spec",
			spec:    nil,
			wantNil: true,
		},
		{
			name:    "no containers",
			spec:    &corev1.PodSpec{},
			wantNil: true,
		},
		{
			name: "single container named main",
			spec: &corev1.PodSpec{
				Containers: []corev1.Container{{Name: commonconsts.MainContainerName}},
			},
			wantName: commonconsts.MainContainerName,
		},
		{
			name: "picks container named main even when not first",
			spec: &corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "istio-proxy"},
					{Name: commonconsts.MainContainerName},
					{Name: "sidecar-frontend"},
				},
			},
			wantName: commonconsts.MainContainerName,
		},
		{
			name: "falls back to first container when none named main",
			spec: &corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "worker"},
					{Name: "sidecar"},
				},
			},
			wantName: "worker",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ResolveMainContainer(tc.spec)
			if tc.wantNil {
				if got != nil {
					t.Fatalf("expected nil, got %+v", got)
				}
				return
			}
			if got == nil {
				t.Fatalf("expected container %q, got nil", tc.wantName)
			}
			if got.Name != tc.wantName {
				t.Fatalf("expected container %q, got %q", tc.wantName, got.Name)
			}
		})
	}
}
