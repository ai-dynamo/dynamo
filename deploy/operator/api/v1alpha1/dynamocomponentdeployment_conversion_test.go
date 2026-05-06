/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package v1alpha1

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	testHubOnlyTemplateName = "hub-only-template-name"
	testModelPVCName        = "model-pvc"
)

func dcdRoundTripFromV1beta1(t *testing.T, src *v1beta1.DynamoComponentDeployment) *v1beta1.DynamoComponentDeployment {
	t.Helper()
	a := &DynamoComponentDeployment{}
	if err := a.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	out := &v1beta1.DynamoComponentDeployment{}
	if err := a.ConvertTo(out); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	return out
}

func assertOnlyKnownDCDAnnotations(t *testing.T, annotations map[string]string) {
	t.Helper()
	for key := range annotations {
		switch key {
		case annDCDSpec, annDCDStatus:
		default:
			t.Fatalf("unexpected annotation %q in %v", key, annotations)
		}
	}
}

func TestDCD_RoundTrip_Empty(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "empty", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "empty",
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_RoundTrip_Minimal(t *testing.T) {
	replicas := int32(3)
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "min", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "min",
				ComponentType: v1beta1.ComponentTypeWorker,
				Replicas:      &replicas,
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_IntermediateHubEditsWinOverPreservedSpoke(t *testing.T) {
	src := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "edit", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ServiceName:   "edit",
				ComponentType: string(v1beta1.ComponentTypeWorker),
			},
		},
	}
	hub := &v1beta1.DynamoComponentDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	hub.Spec.ComponentType = v1beta1.ComponentTypePlanner

	restored := &DynamoComponentDeployment{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if restored.Spec.ComponentType != string(v1beta1.ComponentTypePlanner) {
		t.Fatalf("componentType = %q, want %q", restored.Spec.ComponentType, v1beta1.ComponentTypePlanner)
	}
}

func TestDCD_OmittedServiceNameOriginDoesNotOverrideHubEdit(t *testing.T) {
	src := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "default-name", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			BackendFramework: backendFrameworkSGLang,
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ComponentType: string(v1beta1.ComponentTypeWorker),
			},
		},
	}

	hub := &v1beta1.DynamoComponentDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if hub.Spec.ComponentName != "default-name" {
		t.Fatalf("expected omitted serviceName to default to metadata.name, got %q", hub.Spec.ComponentName)
	}
	assertOnlyKnownDCDAnnotations(t, hub.Annotations)
	raw := hub.Annotations[annDCDSpec]
	if raw == "" {
		t.Fatalf("expected empty serviceName marker in %q, got %v", annDCDSpec, hub.Annotations)
	}
	if _, emptyServiceName, ok := restoreDCDSpokeSpec(raw); !ok || !emptyServiceName {
		t.Fatalf("expected empty serviceName marker in %q payload: %s", annDCDSpec, raw)
	}

	roundTripped := &DynamoComponentDeployment{}
	if err := roundTripped.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if roundTripped.Spec.ServiceName != "" {
		t.Fatalf("expected omitted serviceName to round-trip as empty, got %q", roundTripped.Spec.ServiceName)
	}

	hub.Spec.ComponentName = "live-edit"
	edited := &DynamoComponentDeployment{}
	if err := edited.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom edited hub: %v", err)
	}
	if edited.Spec.ServiceName != "live-edit" {
		t.Fatalf("expected live hub name edit to win over stale origin marker, got %q", edited.Spec.ServiceName)
	}
}

func TestDCD_IntermediateHubPodTemplateEditsRoundTripThroughSpoke(t *testing.T) {
	src := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "hub-only-edit", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ServiceName:   "hub-only-edit",
				ComponentType: string(v1beta1.ComponentTypeWorker),
			},
		},
	}
	hub := &v1beta1.DynamoComponentDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	hub.Spec.PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "worker:edited"}},
		},
	}

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if raw, ok := spoke.Annotations[annDCDSpec]; ok {
		preserved, ok := restoreDCDHubSpec(raw)
		if !ok {
			t.Fatalf("failed to restore %q payload: %s", annDCDSpec, raw)
		}
		if preserved.PodTemplate != nil {
			main, ok := findContainerByName(preserved.PodTemplate.Spec.Containers, mainContainerName)
			if !ok {
				t.Fatalf("expected sparse preserved main-container key, got %#v", preserved.PodTemplate)
			}
			if main.Image != "" {
				t.Fatalf("representable main-container image was preserved: %#v", main)
			}
		}
	}
	if spoke.Spec.ExtraPodSpec == nil || spoke.Spec.ExtraPodSpec.MainContainer == nil {
		t.Fatalf("expected podTemplate main container to be represented in ExtraPodSpec, got %#v", spoke.Spec.ExtraPodSpec)
	}
	if got := spoke.Spec.ExtraPodSpec.MainContainer.Image; got != "worker:edited" {
		t.Fatalf("ExtraPodSpec.MainContainer.Image = %q, want worker:edited", got)
	}

	restored := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if diff := cmp.Diff(hub.Spec.PodTemplate, restored.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch after preserving hub-only edit (-want +got):\n%s", diff)
	}
}

func TestDCD_IntermediateSpokeAlphaOnlyEditsSurvivePreservedHub(t *testing.T) {
	original := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "alpha-only-edit", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "alpha-only-edit",
				ComponentType: v1beta1.ComponentTypeWorker,
			},
		},
	}
	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}

	spoke.Spec.Autoscaling = &Autoscaling{Enabled: true, MinReplicas: 1, MaxReplicas: 3}

	restoredHub := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	restoredSpoke := &DynamoComponentDeployment{}
	if err := restoredSpoke.ConvertFrom(restoredHub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if diff := cmp.Diff(spoke.Spec.Autoscaling, restoredSpoke.Spec.Autoscaling); diff != "" {
		t.Fatalf("autoscaling mismatch after preserving alpha-only edit (-want +got):\n%s", diff)
	}
}

func TestDCD_IntermediateSpokeExtraPodSpecEditsSurvivePreservedHub(t *testing.T) {
	original := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "extra-pod-spec-edit", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "extra-pod-spec-edit",
				ComponentType: v1beta1.ComponentTypeWorker,
			},
		},
	}
	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}

	spoke.Spec.ExtraPodSpec = &ExtraPodSpec{
		MainContainer: &corev1.Container{
			Name:  "custom-main",
			Image: "worker:edited",
		},
	}

	restoredHub := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	restoredSpoke := &DynamoComponentDeployment{}
	if err := restoredSpoke.ConvertFrom(restoredHub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if diff := cmp.Diff(spoke.Spec.ExtraPodSpec, restoredSpoke.Spec.ExtraPodSpec); diff != "" {
		t.Fatalf("extraPodSpec mismatch after preserving alpha-only edit (-want +got):\n%s", diff)
	}
}

func TestDCD_IntermediateHubSharedMemorySizeEditWinsOverPreservedOrigin(t *testing.T) {
	src := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "shared-memory-edit", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				SharedMemory: &SharedMemorySpec{
					Disabled: true,
					Size:     resource.MustParse("1Gi"),
				},
			},
		},
	}
	hub := &v1beta1.DynamoComponentDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	editedSize := resource.MustParse("16Gi")
	hub.Spec.SharedMemorySize = &editedSize

	got := &DynamoComponentDeployment{}
	if err := got.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if got.Spec.SharedMemory == nil {
		t.Fatal("SharedMemory is nil")
	}
	if got.Spec.SharedMemory.Disabled {
		t.Fatalf("SharedMemory.Disabled = true, want false")
	}
	if got.Spec.SharedMemory.Size.Cmp(editedSize) != 0 {
		t.Fatalf("SharedMemory.Size = %s, want %s", got.Spec.SharedMemory.Size.String(), editedSize.String())
	}
}

func TestDCD_RoundTrip_PodTemplate(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pt", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "pt",
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:  "main",
								Image: "dynamo:latest",
								Resources: corev1.ResourceRequirements{
									Requests: corev1.ResourceList{
										corev1.ResourceCPU:                    resource.MustParse("2"),
										corev1.ResourceMemory:                 resource.MustParse("8Gi"),
										corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
									},
								},
							},
							{
								Name:  "logger",
								Image: "fluent/fluent-bit",
							},
						},
					},
				},
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_RoundTrip_PodTemplateKeepsGeneratedMainMarker(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "generated-main-marker", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName:   "generated-main-marker",
				FrontendSidecar: ptr.To(defaultFrontendSidecarContainerName),
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "main"},
							{
								Name:  defaultFrontendSidecarContainerName,
								Image: "frontend:v1",
							},
						},
					},
				},
			},
		},
	}

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	raw := spoke.Annotations[annDCDSpec]
	if raw == "" {
		t.Fatalf("expected %s to preserve generated main marker", annDCDSpec)
	}
	preserved, ok := restoreDCDHubSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDCDSpec, raw)
	}
	main, ok := findContainerByName(preserved.PodTemplate.Spec.Containers, mainContainerName)
	if !ok || !containerHasOnlyName(main) {
		t.Fatalf("expected sparse generated main marker, got %#v", preserved.PodTemplate.Spec.Containers)
	}

	got := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(got); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if diff := cmp.Diff(src.Spec.PodTemplate, got.Spec.PodTemplate, cmpopts.EquateEmpty()); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_RoundTrip_CompilationCacheWithoutGeneratedMountKeepsMarker(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "compilation-cache-no-mount", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "compilation-cache-no-mount",
				CompilationCache: &v1beta1.CompilationCacheConfig{
					PVCName:   "cache-pvc",
					MountPath: "/opt/cache",
				},
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name: "main",
							ReadinessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									Exec: &corev1.ExecAction{Command: []string{"ready"}},
								},
							},
						}},
					},
				},
			},
		},
	}

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	raw := spoke.Annotations[annDCDSpec]
	if raw == "" {
		t.Fatalf("expected %s to preserve absent generated compilation-cache mount", annDCDSpec)
	}
	preserved, ok := restoreDCDHubSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDCDSpec, raw)
	}
	main, ok := findContainerByName(preserved.PodTemplate.Spec.Containers, mainContainerName)
	if !ok || !containerHasOnlyName(main) {
		t.Fatalf("expected sparse main marker for absent generated mount, got %#v", preserved.PodTemplate.Spec.Containers)
	}

	got := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(got); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	gotMain, ok := findContainerByName(got.Spec.PodTemplate.Spec.Containers, mainContainerName)
	if !ok {
		t.Fatalf("expected restored main container, got %#v", got.Spec.PodTemplate.Spec.Containers)
	}
	if len(gotMain.VolumeMounts) != 0 {
		t.Fatalf("generated compilation-cache mount was restored: %#v", gotMain.VolumeMounts)
	}
	if diff := cmp.Diff(src.Spec.PodTemplate, got.Spec.PodTemplate, cmpopts.EquateEmpty()); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_HubSnapshotIsBaseAndV1alpha1OverlayWins(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pt-overlay", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "pt-overlay",
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Name:        testHubOnlyTemplateName,
						Labels:      map[string]string{"old": "label"},
						Annotations: map[string]string{"old": "annotation"},
					},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "worker:old",
							Env:   []corev1.EnvVar{{Name: "OLD", Value: "old"}},
							ReadinessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									Exec: &corev1.ExecAction{Command: []string{"old"}},
								},
							},
							VolumeMounts: []corev1.VolumeMount{{
								Name:      testModelPVCName,
								MountPath: "/old-models",
								ReadOnly:  true,
								SubPath:   "weights",
							}},
						}},
					},
				},
			},
		},
	}

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	spoke.Spec.BackendFramework = backendFrameworkSGLang
	spoke.Spec.ComponentType = string(v1beta1.ComponentTypePlanner)
	spoke.Spec.Envs = []corev1.EnvVar{{Name: "NEW", Value: "new"}}
	spoke.Spec.ExtraPodMetadata = &ExtraPodMetadata{
		Labels:      map[string]string{"new": "label"},
		Annotations: map[string]string{"new": "annotation"},
	}
	spoke.Spec.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{Command: []string{"new"}},
		},
	}
	spoke.Spec.VolumeMounts = []VolumeMount{{Name: testModelPVCName, MountPoint: "/new-models"}}

	got := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(got); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	if got.Spec.BackendFramework != backendFrameworkSGLang {
		t.Fatalf("expected v1alpha1 backendFramework edit to win, got %q", got.Spec.BackendFramework)
	}
	if got.Spec.ComponentType != v1beta1.ComponentTypePlanner {
		t.Fatalf("expected v1alpha1 componentType edit to win, got %q", got.Spec.ComponentType)
	}
	if got.Spec.PodTemplate == nil || got.Spec.PodTemplate.Name != testHubOnlyTemplateName {
		t.Fatalf("expected hub-only podTemplate metadata to be preserved, got %#v", got.Spec.PodTemplate)
	}
	if diff := cmp.Diff(map[string]string{"new": "label"}, got.Spec.PodTemplate.Labels); diff != "" {
		t.Fatalf("expected v1alpha1 podTemplate labels to win (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]string{"new": "annotation"}, got.Spec.PodTemplate.Annotations); diff != "" {
		t.Fatalf("expected v1alpha1 podTemplate annotations to win (-want +got):\n%s", diff)
	}
	main, ok := findContainerByName(got.Spec.PodTemplate.Spec.Containers, "main")
	if !ok {
		t.Fatalf("expected converted podTemplate to include main container, got %#v", got.Spec.PodTemplate.Spec.Containers)
	}
	if main.Image != "worker:old" {
		t.Fatalf("expected hub main-container image to survive through v1alpha1 ExtraPodSpec, got %q", main.Image)
	}
	if diff := cmp.Diff([]corev1.EnvVar{{Name: "NEW", Value: "new"}}, main.Env); diff != "" {
		t.Fatalf("expected v1alpha1 env edit to win (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(spoke.Spec.ReadinessProbe, main.ReadinessProbe); diff != "" {
		t.Fatalf("expected v1alpha1 readinessProbe edit to win (-want +got):\n%s", diff)
	}
	if len(main.VolumeMounts) != 1 {
		t.Fatalf("expected one main volume mount, got %#v", main.VolumeMounts)
	}
	if main.VolumeMounts[0].Name != testModelPVCName || main.VolumeMounts[0].MountPath != "/new-models" {
		t.Fatalf("expected v1alpha1 volume mount name/path to win, got %#v", main.VolumeMounts[0])
	}
	if !main.VolumeMounts[0].ReadOnly || main.VolumeMounts[0].SubPath != "weights" {
		t.Fatalf("expected hub-only volume mount fields to be preserved, got %#v", main.VolumeMounts[0])
	}
}

func TestDCD_RoundTrip_Experimental(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "exp", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "exp",
				ComponentType: v1beta1.ComponentTypeWorker,
				Experimental: &v1beta1.ExperimentalSpec{
					GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{
						Mode:            v1beta1.GMSModeIntraPod,
						DeviceClassName: "gpu.nvidia.com",
					},
				},
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_ExperimentalModeValuesAreValidForIntermediateVersion(t *testing.T) {
	alpha := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "alpha-enums", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				GPUMemoryService: &GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    GMSModeInterPod,
				},
				Failover: &FailoverSpec{
					Enabled: true,
					Mode:    GMSModeIntraPod,
				},
				Checkpoint: &ServiceCheckpointConfig{
					Enabled: true,
					Mode:    CheckpointModeManual,
					Identity: &DynamoCheckpointIdentity{
						Model:            "model",
						BackendFramework: "vllm",
					},
				},
			},
		},
	}
	hub := &v1beta1.DynamoComponentDeployment{}
	if err := alpha.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if got := hub.Spec.Experimental.GPUMemoryService.Mode; got != v1beta1.GMSModeInterPod {
		t.Fatalf("hub GMS mode = %q, want %q", got, v1beta1.GMSModeInterPod)
	}
	if got := hub.Spec.Experimental.Failover.Mode; got != v1beta1.GMSModeIntraPod {
		t.Fatalf("hub failover mode = %q, want %q", got, v1beta1.GMSModeIntraPod)
	}
	if got := hub.Spec.Experimental.Checkpoint.Mode; got != v1beta1.CheckpointModeManual {
		t.Fatalf("hub checkpoint mode = %q, want %q", got, v1beta1.CheckpointModeManual)
	}

	beta := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "beta-enums", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				Experimental: &v1beta1.ExperimentalSpec{
					GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{Mode: v1beta1.GMSModeIntraPod},
					Failover:         &v1beta1.FailoverSpec{Mode: v1beta1.GMSModeInterPod},
					Checkpoint: &v1beta1.ComponentCheckpointConfig{
						Mode: v1beta1.CheckpointModeAuto,
						Identity: &v1beta1.DynamoCheckpointIdentity{
							Model:            "model",
							BackendFramework: "vllm",
						},
					},
				},
			},
		},
	}
	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(beta); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if got := spoke.Spec.GPUMemoryService.Mode; got != GMSModeIntraPod {
		t.Fatalf("spoke GMS mode = %q, want %q", got, GMSModeIntraPod)
	}
	if got := spoke.Spec.Failover.Mode; got != GMSModeInterPod {
		t.Fatalf("spoke failover mode = %q, want %q", got, GMSModeInterPod)
	}
	if got := spoke.Spec.Checkpoint.Mode; got != CheckpointModeAuto {
		t.Fatalf("spoke checkpoint mode = %q, want %q", got, CheckpointModeAuto)
	}
}

// -----------------------------------------------------------------------------
// Expanded DCD coverage: status, v1alpha1-only shapes, scrubbing, JSON bytes.
// -----------------------------------------------------------------------------

// TestDCD_RoundTrip_Status exercises the DCD status fields (conditions and
// single-service replica status).
func TestDCD_RoundTrip_Status(t *testing.T) {
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "status", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "status",
			},
		},
		Status: v1beta1.DynamoComponentDeploymentStatus{
			ObservedGeneration: 4,
			Conditions: []metav1.Condition{
				{Type: "Available", Status: metav1.ConditionTrue, Reason: "AllReady", Message: "ok"},
			},
			Component: &v1beta1.ComponentReplicaStatus{
				ComponentKind:   v1beta1.ComponentKindDeployment,
				ComponentNames:  []string{"dcd-0"},
				Replicas:        3,
				UpdatedReplicas: 3,
				ReadyReplicas:   ptr.To(int32(3)),
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDCD_FromV1alpha1_SparseSpokeSaveCarriesAlphaOnlyFields exercises the
// v1alpha1-only fields preserved through the sparse DCD spoke annotation.
// Fields that flow through podTemplate decomposition (EnvFromSecret,
// Resources, VolumeMounts, Probes) are exercised via the v1beta1-first
// round-trip instead.
func TestDCD_FromV1alpha1_SparseSpokeSaveCarriesAlphaOnlyFields(t *testing.T) {
	dynNs := "legacy-dyn-ns"
	src := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "full", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ComponentType:    "worker",
				SubComponentType: "custom-prefill",
				ServiceName:      "worker-svc",
				DynamoNamespace:  &dynNs,
				Annotations:      map[string]string{"team": "alpha"},
				Labels:           map[string]string{"tier": "gpu"},
				SharedMemory:     &SharedMemorySpec{Disabled: true},
				Autoscaling:      &Autoscaling{Enabled: true, MinReplicas: 1, MaxReplicas: 5},
				Ingress:          &IngressSpec{Enabled: true, Host: "api.example.com"},
				ScalingAdapter:   &ScalingAdapter{Enabled: false},
				GPUMemoryService: &GPUMemoryServiceSpec{Enabled: false, Mode: GMSModeIntraPod},
			},
		},
	}

	b := &v1beta1.DynamoComponentDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	assertOnlyKnownDCDAnnotations(t, b.Annotations)
	raw := b.Annotations[annDCDSpec]
	if raw == "" {
		t.Fatalf("expected sparse spoke save in %q, got %v", annDCDSpec, b.Annotations)
	}
	saved, _, ok := restoreDCDSpokeSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDCDSpec, raw)
	}
	if saved.SubComponentType != "custom-prefill" {
		t.Fatalf("saved SubComponentType = %q, want custom-prefill", saved.SubComponentType)
	}
	if saved.DynamoNamespace == nil || *saved.DynamoNamespace != dynNs {
		t.Fatalf("saved DynamoNamespace = %v, want %q", saved.DynamoNamespace, dynNs)
	}
	if saved.Autoscaling == nil || saved.Ingress == nil || saved.ScalingAdapter == nil || saved.GPUMemoryService == nil {
		t.Fatalf("expected alpha-only fields in sparse save, got %#v", saved)
	}
	if diff := cmp.Diff(src.Spec.Annotations, saved.Annotations); diff != "" {
		t.Fatalf("saved annotations mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(src.Spec.Labels, saved.Labels); diff != "" {
		t.Fatalf("saved labels mismatch (-want +got):\n%s", diff)
	}

	got := &DynamoComponentDeployment{}
	if err := got.ConvertFrom(b); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDCD_FromV1alpha1_PodTemplateDedicatedFields(t *testing.T) {
	src := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-dedicated", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ComponentType: "worker",
				ServiceName:   "pod-dedicated",
				Resources: &Resources{
					Requests: &ResourceItem{CPU: "2", Memory: "4Gi", GPU: "1"},
				},
				VolumeMounts: []VolumeMount{{Name: testModelPVCName, MountPoint: "/models"}},
			},
		},
	}

	b := &v1beta1.DynamoComponentDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	got := &DynamoComponentDeployment{}
	if err := got.ConvertFrom(b); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
	if got.Spec.ExtraPodSpec != nil {
		t.Fatalf("dedicated Resources/VolumeMounts should not come back as ExtraPodSpec: %#v", got.Spec.ExtraPodSpec)
	}
}

func TestDCD_FromV1alpha1_EmptyExtraPodSpecDoesNotMaterializePodTemplate(t *testing.T) {
	src := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "empty-extra", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ComponentType: "worker",
				ServiceName:   "empty-extra",
				ExtraPodSpec:  &ExtraPodSpec{},
			},
		},
	}

	b := &v1beta1.DynamoComponentDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if b.Spec.PodTemplate != nil {
		t.Fatalf("empty ExtraPodSpec should not materialize podTemplate, got %#v", b.Spec.PodTemplate)
	}
	got := &DynamoComponentDeployment{}
	if err := got.ConvertFrom(b); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDCD_JSONRoundTrip_Bytes asserts byte-identical JSON representation
// across a v1beta1 -> v1alpha1 -> v1beta1 round-trip. The PodTemplate carries
// an empty PodTemplateSpec.ObjectMeta and a Container with empty Resources so
// the v1beta1 MarshalJSON normalizer (which strips zero-value
// podTemplate.metadata{} and containers[*].resources{}) is exercised.
func TestDCD_JSONRoundTrip_Bytes(t *testing.T) {
	shm := resource.MustParse("4Gi")
	replicas := int32(2)
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "json", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName:    "json",
				ComponentType:    v1beta1.ComponentTypeWorker,
				Replicas:         &replicas,
				SharedMemorySize: &shm,
				ScalingAdapter:   &v1beta1.ScalingAdapter{},
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{
								Name:      "main",
								Image:     "dynamo:latest",
								Resources: corev1.ResourceRequirements{},
							},
						},
					},
				},
			},
		},
	}
	got := dcdRoundTripFromV1beta1(t, src)

	wantBytes, err := json.Marshal(src)
	if err != nil {
		t.Fatalf("marshal src: %v", err)
	}
	gotBytes, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal got: %v", err)
	}
	if string(wantBytes) != string(gotBytes) {
		t.Errorf("JSON byte-level round-trip mismatch:\nwant: %s\ngot:  %s", wantBytes, gotBytes)
	}
}
