// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"testing"

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
			name: "single container is main by convention",
			spec: &corev1.PodSpec{
				Containers: []corev1.Container{{Name: "main"}},
			},
			wantName: "main",
		},
		{
			name: "picks container named main even when not first",
			spec: &corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "istio-proxy"},
					{Name: "main"},
					{Name: "sidecar-frontend"},
				},
			},
			wantName: "main",
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

func TestRequireExecFormCommand(t *testing.T) {
	for _, tc := range []struct {
		name    string
		cmd     []string
		wantErr bool
	}{
		{"nil command", nil, false},
		{"empty command", []string{}, false},
		{"exec form python", []string{"python3", "-m", "dynamo.vllm"}, false},
		{"exec form /usr/bin/python3", []string{"/usr/bin/python3"}, false},
		{"shell without -c is fine", []string{"/bin/sh", "my-entrypoint.sh"}, false},
		{"bash without -c is fine", []string{"bash", "-l"}, false},
		{"/bin/sh -c rejected", []string{"/bin/sh", "-c"}, true},
		{"sh -c rejected", []string{"sh", "-c"}, true},
		{"bash -c rejected", []string{"/bin/bash", "-c"}, true},
		{"dash -c rejected", []string{"dash", "-c"}, true},
		{"busybox sh -c rejected", []string{"/bin/busybox", "sh", "-c"}, true},
		{"custom binary passes", []string{"/opt/my-runner", "-c", "config.yaml"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := RequireExecFormCommand(&corev1.Container{Name: "main", Command: tc.cmd})
			if tc.wantErr && err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	if err := RequireExecFormCommand(nil); err != nil {
		t.Fatalf("nil container should return nil error, got %v", err)
	}
}

func TestResolveMainContainerName(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{
		Containers: []corev1.Container{
			{Name: "istio-proxy"},
			{Name: "main"},
		},
	}}
	if got := ResolveMainContainerName(pod); got != "main" {
		t.Fatalf("expected main, got %q", got)
	}

	pod = &corev1.Pod{Spec: corev1.PodSpec{
		Containers: []corev1.Container{{Name: "worker"}},
	}}
	if got := ResolveMainContainerName(pod); got != "worker" {
		t.Fatalf("expected worker (fallback), got %q", got)
	}

	pod = &corev1.Pod{}
	if got := ResolveMainContainerName(pod); got != "" {
		t.Fatalf("expected empty string for container-less pod, got %q", got)
	}
}
