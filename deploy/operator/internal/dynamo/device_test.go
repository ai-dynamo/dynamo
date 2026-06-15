/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestApplyDeviceSpec_NoDevice(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main"}},
	}
	applyDeviceSpec(podSpec, nil)
	if podSpec.SchedulerName != "" || len(podSpec.Tolerations) != 0 || len(podSpec.NodeSelector) != 0 {
		t.Fatalf("podSpec changed when device was nil: %+v", podSpec)
	}
	if got := podSpec.Containers[0].Resources.Limits["nvidia.com/gpu"]; !got.IsZero() {
		t.Errorf("unexpected GPU limit set: %s", got.String())
	}
}

func TestApplyDeviceSpec_HAMiShape(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main"}},
	}
	device := &v1beta1.DeviceSpec{
		Resources: corev1.ResourceList{
			"nvidia.com/gpu":    resource.MustParse("1"),
			"nvidia.com/gpumem": resource.MustParse("3000"),
		},
		Tolerations: []corev1.Toleration{
			{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists},
		},
		NodeSelector:  map[string]string{"gpu": "on"},
		SchedulerName: "hami-scheduler",
	}

	applyDeviceSpec(podSpec, device)

	main := podSpec.Containers[0]
	if got := main.Resources.Limits["nvidia.com/gpu"]; got.Cmp(resource.MustParse("1")) != 0 {
		t.Errorf("limits nvidia.com/gpu = %s, want 1", got.String())
	}
	if got := main.Resources.Limits["nvidia.com/gpumem"]; got.Cmp(resource.MustParse("3000")) != 0 {
		t.Errorf("limits nvidia.com/gpumem = %s, want 3000", got.String())
	}
	if got := main.Resources.Requests["nvidia.com/gpu"]; got.Cmp(resource.MustParse("1")) != 0 {
		t.Errorf("requests nvidia.com/gpu = %s, want 1", got.String())
	}
	if got := podSpec.NodeSelector["gpu"]; got != "on" {
		t.Errorf("nodeSelector[gpu] = %q, want on", got)
	}
	if podSpec.SchedulerName != "hami-scheduler" {
		t.Errorf("schedulerName = %q, want hami-scheduler", podSpec.SchedulerName)
	}
	if len(podSpec.Tolerations) != 1 || podSpec.Tolerations[0].Key != "nvidia.com/gpu" {
		t.Errorf("tolerations = %+v, want one nvidia.com/gpu toleration", podSpec.Tolerations)
	}
}

func TestApplyDeviceSpec_AMD(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main"}},
	}
	device := &v1beta1.DeviceSpec{
		Resources: corev1.ResourceList{
			"amd.com/gpu": resource.MustParse("2"),
		},
		Tolerations: []corev1.Toleration{
			{Key: "amd.com/gpu", Operator: corev1.TolerationOpExists},
		},
	}

	applyDeviceSpec(podSpec, device)

	main := podSpec.Containers[0]
	if got := main.Resources.Limits["amd.com/gpu"]; got.Cmp(resource.MustParse("2")) != 0 {
		t.Errorf("limits amd.com/gpu = %s, want 2", got.String())
	}
	if len(podSpec.Tolerations) != 1 || podSpec.Tolerations[0].Key != "amd.com/gpu" {
		t.Errorf("tolerations = %+v, want one amd.com/gpu toleration", podSpec.Tolerations)
	}
}

func TestApplyDeviceSpec_PreservesPodTemplateOverride(t *testing.T) {
	// User-provided podTemplate overrides win over Device defaults for
	// nodeSelector and individual resource keys.
	podSpec := &corev1.PodSpec{
		NodeSelector: map[string]string{"role": "inference"},
		Containers: []corev1.Container{{
			Name: "main",
			Resources: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{
					"amd.com/gpu": resource.MustParse("8"),
				},
			},
		}},
	}
	device := &v1beta1.DeviceSpec{
		Resources: corev1.ResourceList{
			"amd.com/gpu": resource.MustParse("2"),
		},
		NodeSelector: map[string]string{"gpu": "on"},
	}

	applyDeviceSpec(podSpec, device)

	main := podSpec.Containers[0]
	if got := main.Resources.Limits["amd.com/gpu"]; got.Cmp(resource.MustParse("8")) != 0 {
		t.Errorf("podTemplate override lost: amd.com/gpu = %s, want 8", got.String())
	}
	// NodeSelector is merged (union), not replaced.
	if podSpec.NodeSelector["gpu"] != "on" || podSpec.NodeSelector["role"] != "inference" {
		t.Errorf("nodeSelector merged incorrectly: %+v", podSpec.NodeSelector)
	}
}

func TestApplyDeviceSpec_DeduplicatesTolerations(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Tolerations: []corev1.Toleration{
			{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists},
		},
		Containers: []corev1.Container{{Name: "main"}},
	}
	device := &v1beta1.DeviceSpec{
		Tolerations: []corev1.Toleration{
			{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists},
			{Key: "amd.com/gpu", Operator: corev1.TolerationOpExists},
		},
	}

	applyDeviceSpec(podSpec, device)

	if len(podSpec.Tolerations) != 2 {
		t.Fatalf("tolerations len = %d, want 2 (one dedup + one new)", len(podSpec.Tolerations))
	}
}

func TestHasAnyGPUResource(t *testing.T) {
	tests := []struct {
		name string
		req  corev1.ResourceRequirements
		want bool
	}{
		{
			name: "no GPU",
			req: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{"cpu": resource.MustParse("1")},
			},
			want: false,
		},
		{
			name: "nvidia.com/gpu limit",
			req: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")},
			},
			want: true,
		},
		{
			name: "amd.com/gpu limit",
			req: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{"amd.com/gpu": resource.MustParse("1")},
			},
			want: true,
		},
		{
			name: "gpu.intel.com/xe limit",
			req: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{"gpu.intel.com/xe": resource.MustParse("1")},
			},
			want: true,
		},
		{
			name: "arbitrary vendor gpu key",
			req: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{"example.com/gpu": resource.MustParse("2")},
			},
			want: true,
		},
		{
			name: "zero limit ignored",
			req: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{"amd.com/gpu": resource.MustParse("0")},
			},
			want: false,
		},
		{
			name: "only request, no limit",
			req: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{"amd.com/gpu": resource.MustParse("1")},
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := HasAnyGPUResource(tt.req); got != tt.want {
				t.Errorf("HasAnyGPUResource() = %v, want %v", got, tt.want)
			}
		})
	}
}
