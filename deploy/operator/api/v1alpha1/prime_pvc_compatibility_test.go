/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// Prime emits PVC bindings only through extraPodSpec. This guards the alpha
// conversion boundary against duplicating those mounts in realized pod specs.
func TestDGD_PrimePVCBindingsConvertExactlyOnce(t *testing.T) {
	pvcName := "model-cache"
	create := false
	services := map[string]*DynamoComponentDeploymentSharedSpec{}
	for _, service := range []struct {
		name          string
		componentType string
		mountPaths    []string
	}{
		{name: "Frontend", componentType: "frontend", mountPaths: []string{"/model-cache"}},
		{name: "VllmPrefillWorker", componentType: "worker", mountPaths: []string{"/model-cache", "/data"}},
		{name: "VllmDecodeWorker", componentType: "worker", mountPaths: []string{"/model-cache", "/data"}},
	} {
		mounts := make([]corev1.VolumeMount, 0, len(service.mountPaths))
		for _, path := range service.mountPaths {
			mounts = append(mounts, corev1.VolumeMount{Name: pvcName, MountPath: path})
		}
		services[service.name] = &DynamoComponentDeploymentSharedSpec{
			ComponentType: service.componentType,
			ExtraPodSpec: &ExtraPodSpec{
				PodSpec: &corev1.PodSpec{Volumes: []corev1.Volume{{
					Name: pvcName,
					VolumeSource: corev1.VolumeSource{
						PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: pvcName},
					},
				}}},
				MainContainer: &corev1.Container{VolumeMounts: mounts},
			},
		}
	}

	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "prime-pvc", Namespace: "default"},
		Spec: DynamoGraphDeploymentSpec{
			PVCs:     []PVC{{Name: &pvcName, Create: &create}},
			Services: services,
		},
	}
	dst := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	if len(dst.Spec.Components) != len(services) {
		t.Fatalf("converted components = %d, want %d", len(dst.Spec.Components), len(services))
	}
	for _, component := range dst.Spec.Components {
		if component.PodTemplate == nil {
			t.Fatalf("%s podTemplate is nil", component.ComponentName)
		}
		main, ok := findContainerByName(component.PodTemplate.Spec.Containers, "main")
		if !ok {
			t.Fatalf("%s main container is missing", component.ComponentName)
		}
		wantPaths := []string{"/model-cache", "/data"}
		if component.ComponentName == "Frontend" {
			wantPaths = []string{"/model-cache"}
		}
		if len(main.VolumeMounts) != len(wantPaths) {
			t.Fatalf("%s mounts = %d, want %d: %#v", component.ComponentName, len(main.VolumeMounts), len(wantPaths), main.VolumeMounts)
		}
		for _, path := range wantPaths {
			if got := countVolumeMount(main.VolumeMounts, pvcName, path); got != 1 {
				t.Fatalf("%s %s mounts = %d, want 1: %#v", component.ComponentName, path, got, main.VolumeMounts)
			}
		}
		if got := countPVCVolume(component.PodTemplate.Spec.Volumes, pvcName); got != 1 {
			t.Fatalf("%s model-cache volumes = %d, want 1: %#v", component.ComponentName, got, component.PodTemplate.Spec.Volumes)
		}
	}
}

func countVolumeMount(mounts []corev1.VolumeMount, name, path string) int {
	count := 0
	for _, mount := range mounts {
		if mount.Name == name && mount.MountPath == path {
			count++
		}
	}
	return count
}

func countPVCVolume(volumes []corev1.Volume, name string) int {
	count := 0
	for _, volume := range volumes {
		if volume.Name == name && volume.PersistentVolumeClaim != nil && volume.PersistentVolumeClaim.ClaimName == name {
			count++
		}
	}
	return count
}
