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
	"reflect"
	"strings"
	"testing"
	"time"

	jsonpatch "github.com/evanphx/json-patch/v5"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const backendFrameworkSGLang = "sglang"

// roundTripFromV1beta1 converts a v1beta1 DGD to v1alpha1 and back, returning
// the final v1beta1 object. For any valid v1beta1 input V the returned V'
// must equal V (syntactic round-trip invariant).
func roundTripFromV1beta1(t *testing.T, src *v1beta1.DynamoGraphDeployment) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	a := &DynamoGraphDeployment{}
	if err := a.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	out := &v1beta1.DynamoGraphDeployment{}
	if err := a.ConvertTo(out); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	return out
}

// roundTripFromV1alpha1 converts a v1alpha1 DGD to v1beta1 and back. The
// returned object should equal the input for v1alpha1 shapes that survive the
// full round-trip. Services-map ordering is not preserved (set-based equality
// is used by the caller when needed).
func roundTripFromV1alpha1(t *testing.T, src *DynamoGraphDeployment) *DynamoGraphDeployment {
	t.Helper()
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	out := &DynamoGraphDeployment{}
	if err := out.ConvertFrom(b); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	return out
}

func assertOnlyKnownDGDAnnotations(t *testing.T, annotations map[string]string) {
	t.Helper()
	for key := range annotations {
		switch key {
		case annDGDSpec, annDGDStatus:
		default:
			t.Fatalf("unexpected annotation %q in %v", key, annotations)
		}
	}
}

func mustRestoreDGDSpokeServiceSave(t *testing.T, hub *v1beta1.DynamoGraphDeployment, serviceName string) *DynamoComponentDeploymentSharedSpec {
	t.Helper()
	raw := hub.Annotations[annDGDSpec]
	if raw == "" {
		t.Fatalf("expected %q in annotations, got %v", annDGDSpec, hub.Annotations)
	}
	saved, ok := restoreDGDSpokeSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDGDSpec, raw)
	}
	service := saved.Services[serviceName]
	if service == nil {
		t.Fatalf("expected service %q in sparse save, got %#v", serviceName, saved.Services)
	}
	return service
}

func TestDGD_RoundTrip_Empty(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "empty", Namespace: "ns"},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_Minimal(t *testing.T) {
	replicas := int32(2)
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "min", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
					Replicas:      &replicas},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_IntermediateHubEditsWinOverPreservedSpoke(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "edit", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ServiceName:   "worker",
					ComponentType: string(v1beta1.ComponentTypeWorker),
				},
			},
		},
	}
	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	hub.Spec.Components[0].ComponentType = v1beta1.ComponentTypePlanner

	restored := &DynamoGraphDeployment{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if restored.Spec.Services["worker"].ComponentType != string(v1beta1.ComponentTypePlanner) {
		t.Fatalf("componentType = %q, want %q", restored.Spec.Services["worker"].ComponentType, v1beta1.ComponentTypePlanner)
	}
}

func TestDGD_IntermediateHubPodTemplateEditsRoundTripThroughSpoke(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "hub-only-edit", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ServiceName:   "worker",
					ComponentType: string(v1beta1.ComponentTypeWorker),
				},
			},
		},
	}
	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	hub.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "worker:edited"}},
		},
	}

	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if raw, ok := spoke.Annotations[annDGDSpec]; ok {
		preserved, ok := restoreDGDHubSpec(raw)
		if !ok {
			t.Fatalf("failed to restore %q payload: %s", annDGDSpec, raw)
		}
		if len(preserved.Components) != 1 {
			t.Fatalf("expected one preserved component, got %#v", preserved.Components)
		}
		if preserved.Components[0].PodTemplate != nil {
			main, ok := findContainerByName(preserved.Components[0].PodTemplate.Spec.Containers, mainContainerName)
			if !ok {
				t.Fatalf("expected sparse preserved main-container key, got %#v", preserved.Components[0].PodTemplate)
			}
			if main.Image != "" {
				t.Fatalf("representable main-container image was preserved: %#v", main)
			}
		}
	}
	component := spoke.Spec.Services["worker"]
	if component == nil || component.ExtraPodSpec == nil || component.ExtraPodSpec.MainContainer == nil {
		t.Fatalf("expected podTemplate main container to be represented in ExtraPodSpec, got %#v", component)
	}
	if got := component.ExtraPodSpec.MainContainer.Image; got != "worker:edited" {
		t.Fatalf("ExtraPodSpec.MainContainer.Image = %q, want worker:edited", got)
	}

	restored := &v1beta1.DynamoGraphDeployment{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if diff := cmp.Diff(hub.Spec.Components[0].PodTemplate, restored.Spec.Components[0].PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch after preserving hub-only edit (-want +got):\n%s", diff)
	}
}

func TestDGD_IntermediateSpokeAlphaOnlyEditsSurvivePreservedHub(t *testing.T) {
	original := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "alpha-only-edit", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
				},
			},
		},
	}
	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(original); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}

	createTrue := true
	name := "edited-pvc"
	spoke.Spec.PVCs = []PVC{
		{
			Create:           &createTrue,
			Name:             &name,
			StorageClass:     "standard",
			Size:             resource.MustParse("10Gi"),
			VolumeAccessMode: corev1.ReadWriteOnce,
		},
	}

	restoredHub := &v1beta1.DynamoGraphDeployment{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	restoredSpoke := &DynamoGraphDeployment{}
	if err := restoredSpoke.ConvertFrom(restoredHub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if diff := cmp.Diff(spoke.Spec.PVCs, restoredSpoke.Spec.PVCs); diff != "" {
		t.Fatalf("PVCs mismatch after preserving alpha-only edit (-want +got):\n%s", diff)
	}
}

func TestDGD_SpokeSaveCarriesAlphaOnlyFieldsSparsely(t *testing.T) {
	createTrue := true
	name := "models"
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "sparse-spoke-save", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			PVCs: []PVC{{
				Create:           &createTrue,
				Name:             &name,
				StorageClass:     "standard",
				Size:             resource.MustParse("10Gi"),
				VolumeAccessMode: corev1.ReadWriteOnce,
			}},
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(v1beta1.ComponentTypeWorker),
					Autoscaling: &Autoscaling{
						Enabled:     true,
						MinReplicas: 1,
						MaxReplicas: 4,
					},
				},
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	assertOnlyKnownDGDAnnotations(t, hub.Annotations)
	raw := hub.Annotations[annDGDSpec]
	if raw == "" {
		t.Fatalf("expected alpha-only fields in %q, got %v", annDGDSpec, hub.Annotations)
	}
	saved, ok := restoreDGDSpokeSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDGDSpec, raw)
	}
	if diff := cmp.Diff(src.Spec.PVCs, saved.PVCs); diff != "" {
		t.Fatalf("PVC save mismatch (-want +got):\n%s", diff)
	}
	if saved.Services["worker"] == nil || saved.Services["worker"].Autoscaling == nil {
		t.Fatalf("expected autoscaling in sparse spoke save, got %#v", saved.Services)
	}
}

func TestDGD_SpokeSaveCarriesNilServicesSparsely(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "nil-service-save", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"nil-service": nil,
				"worker": {
					ComponentType: string(v1beta1.ComponentTypeWorker),
				},
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	raw := hub.Annotations[annDGDSpec]
	if raw == "" {
		t.Fatalf("expected sparse nil-service save in %q, got %v", annDGDSpec, hub.Annotations)
	}
	saved, ok := restoreDGDSpokeSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDGDSpec, raw)
	}
	if len(saved.Services) != 1 || saved.Services["nil-service"] != nil {
		t.Fatalf("expected only nil-service in sparse save, got %#v", saved.Services)
	}

	restored := &DynamoGraphDeployment{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if _, ok := restored.Spec.Services["nil-service"]; !ok || restored.Spec.Services["nil-service"] != nil {
		t.Fatalf("nil service did not round-trip: %#v", restored.Spec.Services)
	}
}

func TestDGD_IntermediateHubStatusComponentNamesWinOverPreservedSpoke(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "status-edit", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ServiceName:   "worker",
					ComponentType: string(v1beta1.ComponentTypeWorker),
				},
			},
		},
		Status: DynamoGraphDeploymentStatus{
			Services: map[string]ServiceReplicaStatus{
				"worker": {
					ComponentName:  "worker-old",
					ComponentNames: []string{"worker-old"},
				},
			},
		},
	}
	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	status := hub.Status.Components["worker"]
	status.ComponentNames = []string{"worker-new"}
	hub.Status.Components["worker"] = status

	restored := &DynamoGraphDeployment{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if got := restored.Status.Services["worker"].ComponentName; got != "worker-new" {
		t.Fatalf("componentName = %q, want worker-new", got)
	}
}

func TestDGD_RoundTrip_SpecLevelFields(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "spec", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Annotations:      map[string]string{"a": "1"},
			Labels:           map[string]string{"l": "v"},
			BackendFramework: backendFrameworkSGLang,
			Env: []corev1.EnvVar{
				{Name: "FOO", Value: "bar"},
			},
			Restart: &v1beta1.Restart{
				ID: "r1",
				Strategy: &v1beta1.RestartStrategy{
					Type:  v1beta1.RestartStrategyTypeParallel,
					Order: []string{"a", "b"},
				},
			},
			TopologyConstraint: &v1beta1.SpecTopologyConstraint{
				ClusterTopologyName: "default",
				PackDomain:          v1beta1.TopologyDomain("rack"),
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_HubSnapshotIsBaseAndV1alpha1OverlayWins(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "overlay", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
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
		},
	}

	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	spoke.Spec.BackendFramework = backendFrameworkSGLang
	spoke.Spec.Services["worker"].ComponentType = string(v1beta1.ComponentTypePlanner)
	spoke.Spec.Services["worker"].Envs = []corev1.EnvVar{{Name: "NEW", Value: "new"}}
	spoke.Spec.Services["worker"].ExtraPodMetadata = &ExtraPodMetadata{
		Labels:      map[string]string{"new": "label"},
		Annotations: map[string]string{"new": "annotation"},
	}
	spoke.Spec.Services["worker"].ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{Command: []string{"new"}},
		},
	}
	spoke.Spec.Services["worker"].VolumeMounts = []VolumeMount{{Name: testModelPVCName, MountPoint: "/new-models"}}

	got := &v1beta1.DynamoGraphDeployment{}
	if err := spoke.ConvertTo(got); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	if got.Spec.BackendFramework != backendFrameworkSGLang {
		t.Fatalf("expected v1alpha1 backendFramework edit to win, got %q", got.Spec.BackendFramework)
	}
	if len(got.Spec.Components) != 1 || got.Spec.Components[0].ComponentType != v1beta1.ComponentTypePlanner {
		t.Fatalf("expected v1alpha1 componentType edit to win, got %#v", got.Spec.Components)
	}
	if got.Spec.Components[0].PodTemplate == nil || got.Spec.Components[0].PodTemplate.Name != testHubOnlyTemplateName {
		t.Fatalf("expected hub-only podTemplate metadata to be preserved, got %#v", got.Spec.Components[0].PodTemplate)
	}
	if diff := cmp.Diff(map[string]string{"new": "label"}, got.Spec.Components[0].PodTemplate.Labels); diff != "" {
		t.Fatalf("expected v1alpha1 podTemplate labels to win (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]string{"new": "annotation"}, got.Spec.Components[0].PodTemplate.Annotations); diff != "" {
		t.Fatalf("expected v1alpha1 podTemplate annotations to win (-want +got):\n%s", diff)
	}
	main, ok := findContainerByName(got.Spec.Components[0].PodTemplate.Spec.Containers, "main")
	if !ok {
		t.Fatalf("expected converted podTemplate to include main container, got %#v", got.Spec.Components[0].PodTemplate.Spec.Containers)
	}
	if main.Image != "worker:old" {
		t.Fatalf("expected hub main-container image to survive through v1alpha1 ExtraPodSpec, got %q", main.Image)
	}
	if diff := cmp.Diff([]corev1.EnvVar{{Name: "NEW", Value: "new"}}, main.Env); diff != "" {
		t.Fatalf("expected v1alpha1 env edit to win (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(spoke.Spec.Services["worker"].ReadinessProbe, main.ReadinessProbe); diff != "" {
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

func TestDGD_RoundTrip_MultipleServicesOrderStable(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "multi", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "cc-planner", ComponentType: v1beta1.ComponentTypePlanner},
				{ComponentName: "aa-frontend", ComponentType: v1beta1.ComponentTypeFrontend},
				{ComponentName: "bb-worker", ComponentType: v1beta1.ComponentTypeWorker},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_Experimental(t *testing.T) {
	ref := "my-checkpoint"
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "exp", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
					Experimental: &v1beta1.ExperimentalSpec{
						GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{
							Mode:            v1beta1.GMSModeIntraPod,
							DeviceClassName: "gpu.nvidia.com",
						},
						Failover: &v1beta1.FailoverSpec{
							Mode:       v1beta1.GMSModeIntraPod,
							NumShadows: 1,
						},
						Checkpoint: &v1beta1.ComponentCheckpointConfig{
							Mode:          v1beta1.CheckpointModeAuto,
							CheckpointRef: &ref,
						},
					}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_PodTemplate(t *testing.T) {
	shm := resource.MustParse("4Gi")
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pt", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:    "worker",
					ComponentType:    v1beta1.ComponentTypeWorker,
					SharedMemorySize: &shm,
					PodTemplate: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Annotations: map[string]string{"prom.io/scrape": annotationTrue},
							Labels:      map[string]string{"tier": "gpu"},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "main",
									Image: "dynamo:latest",
									Env: []corev1.EnvVar{
										{Name: "DYN_COMPONENT", Value: "worker"},
									},
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:                    resource.MustParse("2"),
											corev1.ResourceMemory:                 resource.MustParse("4Gi"),
											corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
										},
									},
								},
							},
						},
					}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	// corev1.ResourceList equality can be quantity-representation-sensitive;
	// use cmpopts to compare canonical forms.
	opts := cmp.Options{
		cmpopts.EquateEmpty(),
	}
	if diff := cmp.Diff(src, got, opts); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_CompilationCache(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "cc", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
					CompilationCache: &v1beta1.CompilationCacheConfig{
						PVCName:   "cache-pvc",
						MountPath: "/opt/cache",
					},
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name: "main",
									VolumeMounts: []corev1.VolumeMount{
										{Name: "cache-pvc", MountPath: "/opt/cache"},
									},
								},
							},
						},
					}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_ScalingAdapter(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "sa", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:  "worker",
					ComponentType:  v1beta1.ComponentTypeWorker,
					ScalingAdapter: &v1beta1.ScalingAdapter{}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_PVCsPreserved verifies that legacy v1alpha1 PVCs survive
// a v1alpha1 -> v1beta1 -> v1alpha1 round-trip via sparse spec preservation.
func TestDGD_FromV1alpha1_PVCsPreserved(t *testing.T) {
	createTrue := true
	name := testModelPVCName
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			PVCs: []PVC{
				{
					Create:           &createTrue,
					Name:             &name,
					StorageClass:     "standard",
					Size:             resource.MustParse("10Gi"),
					VolumeAccessMode: corev1.ReadWriteOnce,
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_DisabledExperimental verifies that v1alpha1
// GMS/Failover/Checkpoint with Enabled=false and payloads survive the
// round-trip via sparse spec preservation.
func TestDGD_FromV1alpha1_DisabledExperimental(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "disabled", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					GPUMemoryService: &GPUMemoryServiceSpec{
						Enabled:         false,
						Mode:            GMSModeIntraPod,
						DeviceClassName: "gpu.nvidia.com",
					},
					Failover: &FailoverSpec{
						Enabled:    false,
						Mode:       GMSModeIntraPod,
						NumShadows: 1,
					},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_SubComponentType verifies that a subComponentType string
// without a v1beta1 first-class component type survives via sparse preservation.
func TestDGD_FromV1alpha1_SubComponentType(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "sub", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:    "worker",
					SubComponentType: "custom-prefill",
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// -----------------------------------------------------------------------------
// Expanded coverage: status, rich shared-spec fields, pod-template details,
// v1alpha1-only shapes, annotation hygiene, JSON byte-identity.
// -----------------------------------------------------------------------------

// TestDGD_RoundTrip_Status exercises every populated Status sub-struct so that
// the ConvertTo / ConvertFrom status paths are covered (conditions, services
// map, restart, checkpoints, rollingUpdate).
func TestDGD_RoundTrip_Status(t *testing.T) {
	now := metav1.NewTime(metav1.Now().Rfc3339Copy().Time)
	later := metav1.NewTime(now.Time.Add(60 * time.Second))
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "status", Namespace: "ns"},
		Status: v1beta1.DynamoGraphDeploymentStatus{
			ObservedGeneration: 7,
			State:              v1beta1.DGDStateSuccessful,
			Conditions: []metav1.Condition{
				{
					Type:               "Ready",
					Status:             metav1.ConditionTrue,
					Reason:             "AllServicesReady",
					Message:            "all services are ready",
					LastTransitionTime: now,
				},
			},
			Components: map[string]v1beta1.ComponentReplicaStatus{
				"worker": {
					ComponentKind:     v1beta1.ComponentKindDeployment,
					ComponentNames:    []string{"dgd-worker-0", "dgd-worker-1"},
					Replicas:          2,
					UpdatedReplicas:   2,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(2)),
				},
			},
			Restart: &v1beta1.RestartStatus{
				ObservedID: "r-123",
				Phase:      v1beta1.RestartPhaseRestarting,
				InProgress: []string{"worker"},
			},
			Checkpoints: map[string]v1beta1.ComponentCheckpointStatus{
				"worker": {
					CheckpointName: "ckpt-abc",
					IdentityHash:   "sha256:deadbeef",
					Ready:          true,
				},
			},
			RollingUpdate: &v1beta1.RollingUpdateStatus{
				Phase:             v1beta1.RollingUpdatePhaseInProgress,
				StartTime:         &now,
				EndTime:           &later,
				UpdatedComponents: []string{"worker"},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_FullSharedSpec covers every first-class v1beta1 shared-spec
// field that has not been exercised elsewhere (DynamoNamespace is v1alpha1-only
// so it lives in a separate test): GlobalDynamoNamespace, Multinode, ModelRef,
// per-service TopologyConstraint, EPPConfig.
func TestDGD_RoundTrip_FullSharedSpec(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "full", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "epp",
					ComponentType: v1beta1.ComponentTypeEPP,
					EPPConfig: &v1beta1.EPPConfig{
						ConfigMapRef: &corev1.ConfigMapKeySelector{
							LocalObjectReference: corev1.LocalObjectReference{Name: "epp-cfg"},
							Key:                  "config.yaml",
						},
					}},
				{
					ComponentName:         "worker",
					ComponentType:         v1beta1.ComponentTypeWorker,
					GlobalDynamoNamespace: true,
					Multinode:             &v1beta1.MultinodeSpec{NodeCount: 4},
					ModelRef: &v1beta1.ModelReference{
						Name:     "llama-3-70b-instruct",
						Revision: "v1",
					},
					TopologyConstraint: &v1beta1.TopologyConstraint{
						PackDomain: v1beta1.TopologyDomain("rack"),
					}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_PodTemplateProbesAndEnvFrom covers the main-container
// fields that decomposePodTemplate preserves through ExtraPodSpec.MainContainer:
// EnvFrom, LivenessProbe, ReadinessProbe, StartupProbe.
func TestDGD_RoundTrip_PodTemplateProbesAndEnvFrom(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "probes", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "main",
									Image: "dynamo:latest",
									EnvFrom: []corev1.EnvFromSource{
										{
											SecretRef: &corev1.SecretEnvSource{
												LocalObjectReference: corev1.LocalObjectReference{Name: "aws-secret"},
											},
										},
									},
									LivenessProbe: &corev1.Probe{
										ProbeHandler: corev1.ProbeHandler{
											HTTPGet: &corev1.HTTPGetAction{Path: "/healthz", Port: intstrFromInt32(8080)},
										},
										InitialDelaySeconds: 5,
									},
									ReadinessProbe: &corev1.Probe{
										ProbeHandler: corev1.ProbeHandler{
											HTTPGet: &corev1.HTTPGetAction{Path: "/ready", Port: intstrFromInt32(8080)},
										},
									},
									StartupProbe: &corev1.Probe{
										ProbeHandler: corev1.ProbeHandler{
											HTTPGet: &corev1.HTTPGetAction{Path: "/startup", Port: intstrFromInt32(8080)},
										},
										FailureThreshold: 30,
									},
								},
							},
						},
					}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_PodSpecExtras covers the non-main-container PodSpec fields
// that flow through ExtraPodSpec.PodSpec: NodeSelector, Tolerations,
// ServiceAccountName, ImagePullSecrets, Volumes.
func TestDGD_RoundTrip_PodSpecExtras(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "extras", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							NodeSelector:       map[string]string{"node-pool": "gpu"},
							ServiceAccountName: "dynamo-sa",
							ImagePullSecrets:   []corev1.LocalObjectReference{{Name: "ghcr-creds"}},
							Tolerations: []corev1.Toleration{
								{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists, Effect: corev1.TaintEffectNoSchedule},
							},
							Volumes: []corev1.Volume{
								{
									Name: "cache",
									VolumeSource: corev1.VolumeSource{
										EmptyDir: &corev1.EmptyDirVolumeSource{},
									},
								},
							},
							Containers: []corev1.Container{
								{Name: "main", Image: "dynamo:latest"},
							},
						},
					}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_FrontendSidecar starts from v1beta1 (hub) with the
// FrontendSidecar string naming a sidecar container in podTemplate.containers.
func TestDGD_RoundTrip_FrontendSidecar(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "fs", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:   "epp",
					ComponentType:   v1beta1.ComponentTypeEPP,
					FrontendSidecar: ptr.To("sidecar-frontend"),
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{Name: "main", Image: "dynamo:latest"},
								{
									Name:  "sidecar-frontend",
									Image: "dynamo-frontend:latest",
									Args:  []string{"-m", "dynamo.frontend"},
								},
							},
						},
					}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_FrontendSidecarWithoutPodTemplate(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "fs-ref-only", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:   "epp",
					ComponentType:   v1beta1.ComponentTypeEPP,
					FrontendSidecar: ptr.To("sidecar-frontend"),
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_SharedMemoryDisabledZero asserts that an explicit
// size="0" (Disabled=true equivalent) survives. Starts from v1beta1.
func TestDGD_RoundTrip_SharedMemoryDisabledZero(t *testing.T) {
	zero := resource.MustParse("0")
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "shm", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:    "worker",
					ComponentType:    v1beta1.ComponentTypeWorker,
					SharedMemorySize: &zero},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_SharedMemoryEdgeCases covers the two v1alpha1-only
// SharedMemorySpec shapes that need sparse spec preservation to round-trip:
// Disabled=true and the empty struct &SharedMemorySpec{}.
func TestDGD_FromV1alpha1_SharedMemoryEdgeCases(t *testing.T) {
	cases := []struct {
		name string
		shm  *SharedMemorySpec
	}{
		{name: "disabled", shm: &SharedMemorySpec{Disabled: true}},
		{name: "empty", shm: &SharedMemorySpec{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := &DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "shm-" + tc.name, Namespace: "ns"},
				Spec: DynamoGraphDeploymentSpec{
					Services: map[string]*DynamoComponentDeploymentSharedSpec{
						"worker": {
							ComponentType: "worker",
							SharedMemory:  tc.shm,
						},
					},
				},
			}
			got := roundTripFromV1alpha1(t, src)
			if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestDGD_FromV1alpha1_ScalingAdapterDisabled checks that the otherwise-unreachable
// &ScalingAdapter{Enabled:false} shape round-trips via sparse spec preservation.
func TestDGD_FromV1alpha1_ScalingAdapterDisabled(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "sad", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:  "worker",
					ScalingAdapter: &ScalingAdapter{Enabled: false},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_CheckpointDisabled checks that Checkpoint{Enabled:false}
// with a non-trivial payload survives via sparse spec preservation.
func TestDGD_FromV1alpha1_CheckpointDisabled(t *testing.T) {
	ref := "my-ckpt"
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "ckpt-disabled", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Checkpoint: &ServiceCheckpointConfig{
						Enabled:       false,
						Mode:          CheckpointModeAuto,
						CheckpointRef: &ref,
					},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_DynamoNamespaceAndServiceName verifies the two simple
// v1alpha1-only string fields round-trip via sparse spec preservation.
func TestDGD_FromV1alpha1_DynamoNamespaceAndServiceName(t *testing.T) {
	ns := "legacy-dyn-ns"
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "legacy", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:   "worker",
					ServiceName:     "worker-svc",
					DynamoNamespace: &ns,
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_PerServiceAnnotationsAndLabels verifies that v1alpha1
// per-service Annotations/Labels (which target Pod+Service+Ingress in the
// v1alpha1 reconcile model and cannot be faithfully placed in
// podTemplate.metadata alone) are preserved via sparse spec preservation.
func TestDGD_FromV1alpha1_PerServiceAnnotationsAndLabels(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pa", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Annotations:   map[string]string{"team": "alpha"},
					Labels:        map[string]string{"tier": "gpu"},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_AutoscalingAndIngress covers both deprecated blocks.
func TestDGD_FromV1alpha1_AutoscalingAndIngress(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "ai", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Autoscaling: &Autoscaling{
						Enabled:     true,
						MinReplicas: 1,
						MaxReplicas: 5,
					},
					Ingress: &IngressSpec{
						Enabled: true,
						Host:    "api.example.com",
					},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_Resources_ForwardOnly asserts that a v1alpha1 Resources
// struct with a non-default GPUType and Custom keys translates into the
// expected corev1.ResourceList on the v1beta1 side. Full bitwise round-trip
// isn't promised for this shape (v1beta1 -> v1alpha1 folds Resources into
// ExtraPodSpec.MainContainer), but the forward translation is exercised here
// to cover resourcesToNative's GPUType/Custom branches.
func TestDGD_FromV1alpha1_Resources_ForwardOnly(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "res", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Resources: &Resources{
						Requests: &ResourceItem{
							CPU:     "2",
							Memory:  "4Gi",
							GPU:     "2",
							GPUType: "gpu.intel.com/xe",
							Custom:  map[string]string{"example.com/fpga": "1"},
						},
						Limits: &ResourceItem{CPU: "4", Memory: "8Gi"},
					},
				},
			},
		},
	}
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if len(b.Spec.Components) != 1 {
		t.Fatalf("expected 1 component, got %d", len(b.Spec.Components))
	}
	pt := b.Spec.Components[0].PodTemplate
	if pt == nil || len(pt.Spec.Containers) == 0 {
		t.Fatalf("expected main container in podTemplate, got %+v", pt)
	}
	req := pt.Spec.Containers[0].Resources.Requests
	gpu := req[corev1.ResourceName("gpu.intel.com/xe")]
	if gpu.String() != "2" {
		t.Errorf("gpu.intel.com/xe = %q, want %q", gpu.String(), "2")
	}
	fpga := req[corev1.ResourceName("example.com/fpga")]
	if fpga.String() != "1" {
		t.Errorf("example.com/fpga = %q, want %q", fpga.String(), "1")
	}
}

func TestDGD_FromV1alpha1_EmptyResourcesRoundTrip(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "empty-resources", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Resources:     &Resources{Claims: []corev1.ResourceClaim{}},
				},
			},
		},
	}

	got := roundTripFromV1alpha1(t, src)
	if got.Spec.Services["worker"].Resources == nil {
		t.Fatalf("empty Resources pointer did not round-trip: %#v", got.Spec.Services["worker"])
	}
}

func TestDGD_FromV1alpha1_UnrepresentableResourcesRoundTrip(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "unrepresentable-resources", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Resources: &Resources{
						Requests: &ResourceItem{CPU: "not-a-quantity"},
						Claims:   []corev1.ResourceClaim{{Name: "gpu-claim", Request: "gpu"}},
					},
				},
			},
		},
	}

	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src.Spec.Services["worker"].Resources, got.Spec.Services["worker"].Resources, cmpopts.EquateEmpty()); diff != "" {
		t.Fatalf("resources mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_FromV1alpha1_EnvFromSecretRoundTrip(t *testing.T) {
	secret := "worker-secret"
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "env-from-secret", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					EnvFromSecret: &secret,
				},
			},
		},
	}

	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Fatalf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_FromV1alpha1_EmptyExtraPodSpecRoundTrip(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "empty-extra-pod-spec", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					ExtraPodSpec:  &ExtraPodSpec{},
				},
			},
		},
	}

	got := roundTripFromV1alpha1(t, src)
	if got.Spec.Services["worker"].ExtraPodSpec == nil {
		t.Fatalf("empty ExtraPodSpec pointer did not round-trip: %#v", got.Spec.Services["worker"])
	}
}

// TestDGD_ConvertFrom_DuplicateComponentNames asserts that ConvertFrom
// returns an error when the v1beta1 spec.components list has two entries
// with the same componentName, instead of silently overwriting the earlier
// entry on map insertion. The CRD's +listType=map +listMapKey=componentName
// already enforces uniqueness at the API server, but the conversion path is
// also reachable from in-memory unit tests and other code paths that bypass
// CRD validation, so the conversion code defends in depth.
func TestDGD_ConvertFrom_DuplicateComponentNames(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "dup", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "frontend"},
				{ComponentName: "frontend"},
			},
		},
	}
	a := &DynamoGraphDeployment{}
	err := a.ConvertFrom(src)
	if err == nil {
		t.Fatalf("ConvertFrom with duplicate componentName must error, got nil")
	}
	if !strings.Contains(err.Error(), "duplicate component name") {
		t.Errorf("error message should mention duplicate component name, got %q", err.Error())
	}
}

// TestDGD_JSONRoundTrip_Bytes is the strongest form of syntactic equality:
// marshal the v1beta1 input to JSON, round-trip through v1alpha1, marshal the
// result, and require byte-identical output. This catches any nil-vs-empty
// divergence that cmp.Diff+EquateEmpty would collapse.
func TestDGD_JSONRoundTrip_Bytes(t *testing.T) {
	shm := resource.MustParse("4Gi")
	replicas := int32(2)
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "json", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Env:              []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
			Restart:          &v1beta1.Restart{ID: "r1"},
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:    "worker",
					ComponentType:    v1beta1.ComponentTypeWorker,
					Replicas:         &replicas,
					SharedMemorySize: &shm,
					ScalingAdapter:   &v1beta1.ScalingAdapter{},
					Experimental: &v1beta1.ExperimentalSpec{
						GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{
							Mode:            v1beta1.GMSModeIntraPod,
							DeviceClassName: "gpu.nvidia.com",
						},
					}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)

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

// intstrFromInt32 returns an intstr.IntOrString wrapping the given int32 port.
// Kept as a small helper so probe definitions stay compact in the expanded
// round-trip tests.
func intstrFromInt32(v int32) intstr.IntOrString {
	return intstr.FromInt32(v)
}

// TestDGD_FromV1alpha1_FrontendSidecarFullRoundTrip exercises the v1alpha1-first
// FrontendSidecar path: the full FrontendSidecarSpec is saved sparsely on
// ConvertTo and restored on ConvertFrom while v1beta1 carries only the name
// reference.
func TestDGD_FromV1alpha1_FrontendSidecarFullRoundTrip(t *testing.T) {
	secret := "frontend-secret"
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "fs-full", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"epp": {
					ComponentType: "epp",
					FrontendSidecar: &FrontendSidecarSpec{
						Image:         "dynamo-frontend:1.2.3",
						Args:          []string{"-m", "dynamo.frontend", "--router-mode", "direct"},
						EnvFromSecret: &secret,
						Envs: []corev1.EnvVar{
							{Name: "FRONTEND_FLAG", Value: annotationTrue},
						},
					},
				},
			},
		},
	}

	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if len(b.Spec.Components) != 1 {
		t.Fatalf("expected 1 component on v1beta1, got %d", len(b.Spec.Components))
	}
	comp := b.Spec.Components[0]
	if comp.FrontendSidecar == nil || *comp.FrontendSidecar != "sidecar-frontend" {
		t.Fatalf("expected FrontendSidecar pointer = %q, got %v", "sidecar-frontend", comp.FrontendSidecar)
	}
	if comp.PodTemplate == nil {
		t.Fatalf("expected podTemplate to carry the sidecar container")
	}
	var sidecar *corev1.Container
	for i := range comp.PodTemplate.Spec.Containers {
		if comp.PodTemplate.Spec.Containers[i].Name == "sidecar-frontend" {
			sidecar = &comp.PodTemplate.Spec.Containers[i]
			break
		}
	}
	if sidecar == nil {
		t.Fatalf("expected 'sidecar-frontend' container in podTemplate, got %+v", comp.PodTemplate.Spec.Containers)
	}
	if sidecar.Image != "dynamo-frontend:1.2.3" {
		t.Errorf("sidecar image: got %q, want %q", sidecar.Image, "dynamo-frontend:1.2.3")
	}
	assertOnlyKnownDGDAnnotations(t, b.Annotations)
	saved := mustRestoreDGDSpokeServiceSave(t, b, "epp")
	if saved.FrontendSidecar == nil {
		t.Fatalf("expected frontendSidecar in sparse spoke save, got %#v", saved)
	}

	got := &DynamoGraphDeployment{}
	if err := got.ConvertFrom(b); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("FrontendSidecar round-trip mismatch (-want +got):\n%s", diff)
	}
	assertOnlyKnownDGDAnnotations(t, got.Annotations)
}

// TestDGD_FromV1alpha1_GMSEnabledFalseEmptyPayload targets the
// "Enabled=false with zero-valued payload -> sparse save" branch in
// convertExperimentalTo for GPUMemoryService. The v1alpha1 pointer
// &GPUMemoryServiceSpec{} (no Mode, no DeviceClassName) must round-trip
// through sparse spec preservation without being collapsed to nil.
func TestDGD_FromV1alpha1_GMSEnabledFalseEmptyPayload(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "gms-empty", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:    "worker",
					GPUMemoryService: &GPUMemoryServiceSpec{Enabled: false},
				},
			},
		},
	}
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	assertOnlyKnownDGDAnnotations(t, b.Annotations)
	saved := mustRestoreDGDSpokeServiceSave(t, b, "worker")
	if saved.GPUMemoryService == nil || saved.GPUMemoryService.Enabled {
		t.Fatalf("expected disabled GPUMemoryService in sparse save, got %#v", saved.GPUMemoryService)
	}

	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("GMS empty-payload round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_FailoverEnabledFalseEmptyPayload targets the
// sibling branch for Failover.
func TestDGD_FromV1alpha1_FailoverEnabledFalseEmptyPayload(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "fo-empty", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Failover:      &FailoverSpec{Enabled: false},
				},
			},
		},
	}
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	assertOnlyKnownDGDAnnotations(t, b.Annotations)
	saved := mustRestoreDGDSpokeServiceSave(t, b, "worker")
	if saved.Failover == nil || saved.Failover.Enabled {
		t.Fatalf("expected disabled Failover in sparse save, got %#v", saved.Failover)
	}

	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("Failover empty-payload round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_CheckpointEnabledFalseEmptyPayload covers the same
// "Enabled=false with zero-valued payload -> sparse save" branch for the
// Checkpoint sibling in convertExperimentalTo.
func TestDGD_FromV1alpha1_CheckpointEnabledFalseEmptyPayload(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "ckpt-empty", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Checkpoint:    &ServiceCheckpointConfig{Enabled: false},
				},
			},
		},
	}
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	assertOnlyKnownDGDAnnotations(t, b.Annotations)
	saved := mustRestoreDGDSpokeServiceSave(t, b, "worker")
	if saved.Checkpoint == nil || saved.Checkpoint.Enabled {
		t.Fatalf("expected disabled Checkpoint in sparse save, got %#v", saved.Checkpoint)
	}

	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("Checkpoint empty-payload round-trip mismatch (-want +got):\n%s", diff)
	}
}

// newIdempotenceDGDFixture returns a representative v1beta1 DGD covering the
// shape that originally surfaced the generation-bump regression in-cluster:
// an aggregated frontend+worker service pair where the frontend has no
// explicit container resources (so `containers[*].resources` projects as an
// empty object) and `sharedMemorySize="0"` (which exercises the Disabled=true
// path in convertSharedMemoryFrom). It is shared by the idempotence tests
// below so the linter does not flag the identical fixture builders as dupl.
func newIdempotenceDGDFixture() *v1beta1.DynamoGraphDeployment {
	replicas := int32(1)
	return &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "conv-smoke", Namespace: "jsm"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:    "frontend",
					ComponentType:    v1beta1.ComponentTypeFrontend,
					Replicas:         &replicas,
					FrontendSidecar:  ptr.To("sidecar-frontend"),
					SharedMemorySize: ptr.To(resource.MustParse("0")),
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{Name: "main", Image: "nvcr.io/nvidia/dynamo:latest"},
								{Name: "sidecar-frontend", Image: "nvcr.io/nvidia/dynamo-frontend:latest"},
							},
						},
					}},
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
					Replicas:      &replicas,
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:  "main",
								Image: "nvcr.io/nvidia/dynamo:latest",
								Resources: corev1.ResourceRequirements{
									Limits: corev1.ResourceList{
										"nvidia.com/gpu": resource.MustParse("1"),
									},
								},
							}},
						},
					}},
			},
		},
	}
}

// TestDGD_ApplyIdempotence_GenerationBump simulates the server-side flow that
// kubectl apply drives on every invocation: the v1beta1 payload is converted
// to v1alpha1 for storage, stored (i.e. JSON-marshaled and unmarshaled), and
// then a second apply of the same v1beta1 payload runs ConvertFrom again.
//
// The API server increments `.metadata.generation` only when reflect.DeepEqual
// reports that oldSpec and newSpec differ. A resource.Quantity that was
// populated by Go zero-initialization is NOT reflect.DeepEqual to the same
// numeric value after a JSON round-trip (the latter carries canonical Format
// / cached-string state that the former lacks), so a ConvertFrom that emits
// bare Quantity{} values produces a spec that churns on every apply even when
// the user-visible bytes are identical. This test pins the invariant that
// ConvertFrom's output must be reflect.DeepEqual-stable across an etcd JSON
// round-trip, so kubectl apply is idempotent for any v1beta1 input.
func TestDGD_ApplyIdempotence_GenerationBump(t *testing.T) {
	// sharedMemorySize=\"0\" is the critical trigger: it exercises the
	// Disabled=true path in convertSharedMemoryFrom, which previously left
	// SharedMemorySpec.Size as a bare Quantity{}.
	newSrc := newIdempotenceDGDFixture

	// First apply: simulates the initial create path.
	stored := &DynamoGraphDeployment{}
	if err := stored.ConvertFrom(newSrc()); err != nil {
		t.Fatalf("first ConvertFrom: %v", err)
	}

	// Simulate etcd: marshal to JSON (what the API server does before handing
	// the object to the storage layer) and unmarshal back into a fresh
	// v1alpha1 value. This canonicalizes any embedded resource.Quantity.
	data, err := json.Marshal(stored)
	if err != nil {
		t.Fatalf("marshal stored: %v", err)
	}
	storedAfterEtcd := &DynamoGraphDeployment{}
	if err := json.Unmarshal(data, storedAfterEtcd); err != nil {
		t.Fatalf("unmarshal stored: %v", err)
	}

	// Second apply: the server materializes the same v1beta1 spec and runs
	// ConvertFrom again; the result must DeepEqual the post-etcd form, or
	// the API server will bump .metadata.generation.
	reapplied := &DynamoGraphDeployment{}
	if err := reapplied.ConvertFrom(newSrc()); err != nil {
		t.Fatalf("second ConvertFrom: %v", err)
	}

	if !reflect.DeepEqual(storedAfterEtcd.Spec, reapplied.Spec) {
		t.Errorf("second apply is not DeepEqual to stored form after JSON round-trip; kubectl apply will bump generation on every invocation.\nstored.spec = %#v\nreapplied.spec = %#v", storedAfterEtcd.Spec, reapplied.Spec)
	}

	if !reflect.DeepEqual(storedAfterEtcd.Annotations, reapplied.Annotations) {
		t.Errorf("annotations drift between applies.\nstored = %#v\nreapplied = %#v", storedAfterEtcd.Annotations, reapplied.Annotations)
	}
}

// TestDGD_ApplyIdempotence_EmptySharedMemoryOrigin pins the twin invariant for
// the `&SharedMemorySpec{}` -> v1beta1 sparse-save path. The empty
// struct has `Size: resource.Quantity{}` which serializes to "0" (Quantity is
// a non-pointer struct, so encoding/json's omitempty does not drop it); after
// the etcd JSON round-trip the Size becomes a canonical zero Quantity that is
// not reflect.DeepEqual to the Go zero value. Without the fix in
// convertSharedMemoryFrom, every reapply of a v1beta1 object carrying the
// sparse save would bump .metadata.generation.
func TestDGD_ApplyIdempotence_EmptySharedMemoryOrigin(t *testing.T) {
	// Seed a v1alpha1 object whose only non-default bit is SharedMemory =
	// &SharedMemorySpec{}, then run it through ConvertTo once so the
	// produced v1beta1 carries the sparse save we need to exercise the empty
	// branch of convertSharedMemoryFrom.
	a1 := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "shm-empty", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					SharedMemory:  &SharedMemorySpec{},
				},
			},
		},
	}
	b1 := &v1beta1.DynamoGraphDeployment{}
	if err := a1.ConvertTo(b1); err != nil {
		t.Fatalf("seed ConvertTo: %v", err)
	}
	// Sanity: the sparse spoke save is what triggers the path we want to
	// cover; fail loudly if future refactors break the assumption.
	assertOnlyKnownDGDAnnotations(t, b1.Annotations)
	saved := mustRestoreDGDSpokeServiceSave(t, b1, "worker")
	if saved.SharedMemory == nil {
		t.Fatalf("expected sharedMemory in sparse spoke save, got: %#v", saved)
	}

	// First apply: ConvertFrom takes the empty branch and produces the
	// v1alpha1 object that the API server will store in etcd.
	stored := &DynamoGraphDeployment{}
	if err := stored.ConvertFrom(b1); err != nil {
		t.Fatalf("first ConvertFrom: %v", err)
	}

	data, err := json.Marshal(stored)
	if err != nil {
		t.Fatalf("marshal stored: %v", err)
	}
	storedAfterEtcd := &DynamoGraphDeployment{}
	if err := json.Unmarshal(data, storedAfterEtcd); err != nil {
		t.Fatalf("unmarshal stored: %v", err)
	}

	// Second apply of the same v1beta1 payload must produce a spec that is
	// reflect.DeepEqual to the stored-and-reloaded form, or the API server
	// will bump .metadata.generation on every apply.
	reapplied := &DynamoGraphDeployment{}
	if err := reapplied.ConvertFrom(b1); err != nil {
		t.Fatalf("second ConvertFrom: %v", err)
	}

	if !reflect.DeepEqual(storedAfterEtcd.Spec, reapplied.Spec) {
		t.Errorf("empty-SharedMemorySpec reapply is not DeepEqual to post-etcd form; kubectl apply will bump generation on every invocation.\nstored.spec = %#v\nreapplied.spec = %#v", storedAfterEtcd.Spec, reapplied.Spec)
	}
}

// TestDGD_ApplyIdempotence_CSAMergePatch reproduces the *exact* kubectl
// client-side-apply flow the API server drives on every `kubectl apply`
// against the v1beta1 endpoint. The JSON merge patch step is what strips
// `podTemplate.metadata: {}` and `containers[*].resources: {}` from the
// v1beta1 projection before ConvertFrom runs, so the post-merge v1beta1 can
// differ structurally from the pre-merge v1beta1; any asymmetry between the
// converted ExtraPodSpec shape and the stored ExtraPodSpec shape will drift
// `.spec` and cause `.metadata.generation` to bump on every re-apply. The
// `v1beta1.MarshalJSON` normalizer neutralizes that asymmetry by stripping
// the `{}` artefacts before they hit the wire; this test is the regression
// guard against a future change that re-introduces them.
//
// The patch body below is the literal bytes captured from `kubectl apply -v=9`
// applying a representative DGD fixture that uses an aggregated (frontend +
// worker) service pair with no explicit `podTemplate.metadata` or container
// resources on the frontend container -- the shape that originally surfaced
// the generation-bump regression in-cluster.
func TestDGD_ApplyIdempotence_CSAMergePatch(t *testing.T) {
	userB1 := newIdempotenceDGDFixture

	// This is the literal patch body kubectl client-side-apply sends for
	// the fixture above, captured with `kubectl apply -v=9` at the v1beta1
	// endpoint. Arrays are replaced wholesale by JSON merge patch (RFC
	// 7396), so every reapply strips the `podTemplate.metadata: {}`
	// and `containers[*].resources: {}` fields that the server's v1beta1
	// projection adds.
	csaPatch := []byte(`{"spec":{"services":[` +
		`{"componentType":"frontend","frontendSidecar":"sidecar-frontend","name":"frontend","podTemplate":{"spec":{"containers":[{"image":"nvcr.io/nvidia/dynamo:latest","name":"main"},{"image":"nvcr.io/nvidia/dynamo-frontend:latest","name":"sidecar-frontend"}]}},"replicas":1,"sharedMemorySize":"0"},` +
		`{"componentType":"worker","name":"worker","podTemplate":{"spec":{"containers":[{"image":"nvcr.io/nvidia/dynamo:latest","name":"main","resources":{"limits":{"nvidia.com/gpu":"1"}}}]}},"replicas":1}` +
		`]}}`)

	// Step 1: first apply. Server ConvertsFrom the user's v1beta1 to
	// produce the stored v1alpha1 object.
	stored := &DynamoGraphDeployment{}
	if err := stored.ConvertFrom(userB1()); err != nil {
		t.Fatalf("first ConvertFrom (create): %v", err)
	}
	storedBytes, err := json.Marshal(stored)
	if err != nil {
		t.Fatalf("marshal stored: %v", err)
	}
	storedAfterEtcd := &DynamoGraphDeployment{}
	if err := json.Unmarshal(storedBytes, storedAfterEtcd); err != nil {
		t.Fatalf("unmarshal stored: %v", err)
	}

	// Step 2: second apply. Server reads stored, converts to v1beta1 so
	// the JSON merge patch can be applied at the request version.
	serverB1 := &v1beta1.DynamoGraphDeployment{}
	if err := storedAfterEtcd.ConvertTo(serverB1); err != nil {
		t.Fatalf("server ConvertTo: %v", err)
	}
	serverB1Bytes, err := json.Marshal(serverB1)
	if err != nil {
		t.Fatalf("marshal serverB1: %v", err)
	}

	// Step 3: apply JSON merge patch. Arrays are replaced wholesale.
	patchedB1Bytes, err := jsonpatch.MergePatch(serverB1Bytes, csaPatch)
	if err != nil {
		t.Fatalf("merge patch: %v", err)
	}
	patchedB1 := &v1beta1.DynamoGraphDeployment{}
	if err := json.Unmarshal(patchedB1Bytes, patchedB1); err != nil {
		t.Fatalf("unmarshal patched: %v", err)
	}

	// Step 4: server converts back to v1alpha1 for storage. This is
	// `new` in PrepareForUpdate(new, old).
	next := &DynamoGraphDeployment{}
	if err := next.ConvertFrom(patchedB1); err != nil {
		t.Fatalf("server ConvertFrom (write): %v", err)
	}

	// Step 5: apiserver generation-bump check.
	// The apiserver compares old["spec"] vs new["spec"] as unstructured
	// maps (apiequality.Semantic.DeepEqual). We model that by marshalling
	// both sides to JSON and comparing the byte forms, which is what the
	// apiserver effectively does after a JSON round-trip of the webhook
	// response.
	storedJSON, _ := json.Marshal(storedAfterEtcd.Spec)
	nextJSON, _ := json.Marshal(next.Spec)
	if string(storedJSON) != string(nextJSON) {
		t.Errorf("CSA merge-patch flow drifted spec; kubectl apply will bump .metadata.generation on every invocation.\nstored=%s\nnext  =%s", storedJSON, nextJSON)
	}
	if diff := cmp.Diff(storedAfterEtcd.Spec, next.Spec); diff != "" {
		t.Errorf("CSA merge-patch flow drifted spec (Go-level):\n(-stored +next):\n%s", diff)
	}
	if !reflect.DeepEqual(storedAfterEtcd.Annotations, next.Annotations) {
		t.Errorf("annotations drift across CSA merge-patch flow:\nstored=%#v\nnext  =%#v", storedAfterEtcd.Annotations, next.Annotations)
	}

	// Step 6: mimic the apiserver's unstructured-level comparison. The
	// generation bump in PrepareForUpdate is driven by
	// apiequality.Semantic.DeepEqual on the `spec` subtree decoded as
	// map[string]interface{}. Byte-equal JSON is sufficient for that to
	// match, but Go's json.Marshal and apimachinery's unstructured decoder
	// use different field ordering conventions; re-decoding via the same
	// path the apiserver uses surfaces any residual structural drift.
	var oldMap, newMap map[string]interface{}
	if err := json.Unmarshal(storedJSON, &oldMap); err != nil {
		t.Fatalf("unmarshal stored spec: %v", err)
	}
	if err := json.Unmarshal(nextJSON, &newMap); err != nil {
		t.Fatalf("unmarshal next spec: %v", err)
	}
	if !reflect.DeepEqual(oldMap, newMap) {
		t.Errorf("unstructured spec DeepEqual mismatch after CSA merge-patch flow:\nold=%#v\nnew=%#v", oldMap, newMap)
	}
}
