/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func addGeneratedFrontendSidecarHubOnlySecurityContext(t *testing.T, podTemplate *corev1.PodTemplateSpec) {
	t.Helper()
	if podTemplate == nil {
		t.Fatalf("expected generated frontend sidecar podTemplate, got nil")
	}
	for i := range podTemplate.Spec.Containers {
		if podTemplate.Spec.Containers[i].Name != defaultFrontendSidecarContainerName {
			continue
		}
		podTemplate.Spec.Containers[i].SecurityContext = &corev1.SecurityContext{
			RunAsNonRoot: ptr.To(true),
		}
		return
	}
	t.Fatalf("expected generated frontend sidecar in podTemplate, got %#v", podTemplate)
}

func TestBugDCD_HubSidecarOnlyPodTemplateRoundTrips(t *testing.T) {
	in := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "sidecar-only", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "sidecar-only",
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						NodeSelector: map[string]string{"accelerator": "h100"},
						Containers: []corev1.Container{{
							Name:  "metrics",
							Image: "busybox:1.36",
							Args:  []string{"sh", "-c", "sleep 3600"},
						}},
					},
				},
			},
		},
	}

	out := dcdRoundTripFromV1beta1(t, in)
	if diff := cmp.Diff(in.Spec.PodTemplate, out.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDCD_HubEmptyPodTemplateRoundTrips(t *testing.T) {
	in := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "empty-pod-template", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "empty-pod-template",
				PodTemplate:   &corev1.PodTemplateSpec{},
			},
		},
	}

	out := dcdRoundTripFromV1beta1(t, in)
	if diff := cmp.Diff(in.Spec.PodTemplate, out.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDGD_HubEmptyPodTemplateRoundTrips(t *testing.T) {
	in := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "empty-pod-template", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "empty-pod-template",
				PodTemplate:   &corev1.PodTemplateSpec{},
			}},
		},
	}

	out := roundTripFromV1beta1(t, in)
	if diff := cmp.Diff(in.Spec.Components[0].PodTemplate, out.Spec.Components[0].PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDGD_HubOriginSpokeMainContainerEnvSplitRoundTrips(t *testing.T) {
	hub := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "main-container-env-split", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Name: "hub-origin-template"},
				},
			}},
		},
	}

	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom hub: %v", err)
	}
	component := spoke.Spec.Services["worker"]
	if component == nil {
		t.Fatalf("expected worker service after ConvertFrom: %#v", spoke.Spec.Services)
	}

	// Stage 2 edit: the v1alpha1 object intentionally splits env between the
	// flat Envs field and the ExtraPodSpec main-container escape hatch.
	component.Envs = []corev1.EnvVar{{Name: "FLAT", Value: "flat"}}
	component.ExtraPodSpec = &ExtraPodSpec{
		MainContainer: &corev1.Container{
			Env: []corev1.EnvVar{{Name: "EXTRA", Value: "extra"}},
		},
	}

	restoredHub := &v1beta1.DynamoGraphDeployment{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo spoke: %v", err)
	}
	restoredSpoke := &DynamoGraphDeployment{}
	if err := restoredSpoke.ConvertFrom(restoredHub); err != nil {
		t.Fatalf("ConvertFrom restored hub: %v", err)
	}
	restoredComponent := restoredSpoke.Spec.Services["worker"]
	if restoredComponent == nil {
		t.Fatalf("expected restored worker service: %#v", restoredSpoke.Spec.Services)
	}
	if diff := cmp.Diff(component.Envs, restoredComponent.Envs); diff != "" {
		t.Fatalf("flat env origin mismatch (-want +got):\n%s", diff)
	}
	if component.ExtraPodSpec == nil || component.ExtraPodSpec.MainContainer == nil {
		t.Fatalf("test setup lost extra main-container env: %#v", component.ExtraPodSpec)
	}
	if restoredComponent.ExtraPodSpec == nil || restoredComponent.ExtraPodSpec.MainContainer == nil {
		t.Fatalf("extra main-container env collapsed into flat envs: %#v", restoredComponent)
	}
	if diff := cmp.Diff(component.ExtraPodSpec.MainContainer.Env, restoredComponent.ExtraPodSpec.MainContainer.Env); diff != "" {
		t.Fatalf("extra main-container env origin mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDCD_HubPodTemplateWithoutContainersRoundTrips(t *testing.T) {
	in := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-template-no-containers", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "pod-template-no-containers",
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						NodeSelector: map[string]string{"accelerator": "h100"},
					},
				},
			},
		},
	}

	out := dcdRoundTripFromV1beta1(t, in)
	if diff := cmp.Diff(in.Spec.PodTemplate, out.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDCD_HubPodTemplateContainerOrderRoundTrips(t *testing.T) {
	in := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "container-order", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "container-order",
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "metrics", Image: "busybox:1.36"},
							{Name: "main", Image: "dynamo:latest"},
						},
					},
				},
			},
		},
	}

	out := dcdRoundTripFromV1beta1(t, in)
	if diff := cmp.Diff(in.Spec.PodTemplate, out.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDCD_MetadataOnlyPodTemplateSaveDoesNotFreezeContainerOrder(t *testing.T) {
	hub := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "metadata-only-container-order", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "metadata-only-container-order",
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Name: testHubOnlyTemplateName},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "main", Image: "worker:latest"},
							{Name: "metrics", Image: "metrics:latest"},
							{Name: "logger", Image: "logger:latest"},
						},
					},
				},
			},
		},
	}

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	raw := spoke.Annotations[annDCDSpec]
	if raw == "" {
		t.Fatalf("expected %s to preserve hub-only podTemplate metadata", annDCDSpec)
	}
	preserved, ok := restoreDCDHubSpec(raw)
	if !ok {
		t.Fatalf("failed to decode %s: %s", annDCDSpec, raw)
	}
	if preserved.PodTemplate == nil || preserved.PodTemplate.Name != testHubOnlyTemplateName {
		t.Fatalf("expected preserved podTemplate metadata, got %#v", preserved.PodTemplate)
	}
	if len(preserved.PodTemplate.Spec.Containers) != 0 {
		t.Fatalf("metadata-only podTemplate save preserved container order: %#v", preserved.PodTemplate.Spec.Containers)
	}

	if spoke.Spec.ExtraPodSpec == nil ||
		spoke.Spec.ExtraPodSpec.PodSpec == nil ||
		len(spoke.Spec.ExtraPodSpec.PodSpec.Containers) != 2 {
		t.Fatalf("expected two spoke sidecars, got %#v", spoke.Spec.ExtraPodSpec)
	}
	spoke.Spec.ExtraPodSpec.PodSpec.Containers[0], spoke.Spec.ExtraPodSpec.PodSpec.Containers[1] =
		spoke.Spec.ExtraPodSpec.PodSpec.Containers[1], spoke.Spec.ExtraPodSpec.PodSpec.Containers[0]

	restored := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	gotOrder := make([]string, 0, len(restored.Spec.PodTemplate.Spec.Containers))
	for _, container := range restored.Spec.PodTemplate.Spec.Containers {
		gotOrder = append(gotOrder, container.Name)
	}
	wantOrder := []string{"main", "logger", "metrics"}
	if diff := cmp.Diff(wantOrder, gotOrder); diff != "" {
		t.Fatalf("container order mismatch after live spoke reorder (-want +got):\n%s", diff)
	}
	if restored.Spec.PodTemplate.Name != testHubOnlyTemplateName {
		t.Fatalf("podTemplate name = %q, want %s", restored.Spec.PodTemplate.Name, testHubOnlyTemplateName)
	}
}

func TestBugDCD_HubEditToEnvFromSecretOptionalRoundTrips(t *testing.T) {
	alpha := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "env-from-secret", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ServiceName:   "env-from-secret",
				EnvFromSecret: ptr.To("secret-a"),
			},
		},
	}

	hub := &v1beta1.DynamoComponentDeployment{}
	if err := alpha.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	hub.Spec.PodTemplate.Spec.Containers[0].EnvFrom[0].SecretRef.Optional = ptr.To(true)

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	out := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(out); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if diff := cmp.Diff(hub.Spec.PodTemplate, out.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDCD_HubEditToFrontendSidecarHubOnlyFieldRoundTrips(t *testing.T) {
	alpha := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "frontend-sidecar-hub-only", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ServiceName: "frontend-sidecar-hub-only",
				FrontendSidecar: &FrontendSidecarSpec{
					Image: "frontend:v1",
				},
			},
		},
	}

	hub := &v1beta1.DynamoComponentDeployment{}
	if err := alpha.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	foundSidecar := false
	for i := range hub.Spec.PodTemplate.Spec.Containers {
		if hub.Spec.PodTemplate.Spec.Containers[i].Name == defaultFrontendSidecarContainerName {
			foundSidecar = true
			hub.Spec.PodTemplate.Spec.Containers[i].SecurityContext = &corev1.SecurityContext{
				RunAsNonRoot: ptr.To(true),
			}
		}
	}
	if !foundSidecar {
		t.Fatalf("expected generated frontend sidecar in hub podTemplate, got %#v", hub.Spec.PodTemplate)
	}

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	raw, ok := spoke.Annotations[annDCDSpec]
	if !ok {
		t.Fatalf("expected sparse hub preservation in %q, got %v", annDCDSpec, spoke.Annotations)
	}
	preserved, ok := restoreDCDHubSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDCDSpec, raw)
	}
	preservedSidecar, ok := findContainerByName(preserved.PodTemplate.Spec.Containers, defaultFrontendSidecarContainerName)
	if !ok {
		t.Fatalf("expected preserved frontend sidecar key, got %#v", preserved.PodTemplate)
	}
	if preservedSidecar.Image != "" || len(preservedSidecar.Args) > 0 || len(preservedSidecar.Env) > 0 || len(preservedSidecar.EnvFrom) > 0 {
		t.Fatalf("expected sparse frontend sidecar save, got %#v", preservedSidecar)
	}
	if preservedSidecar.SecurityContext == nil || preservedSidecar.SecurityContext.RunAsNonRoot == nil || !*preservedSidecar.SecurityContext.RunAsNonRoot {
		t.Fatalf("expected preserved hub-only securityContext, got %#v", preservedSidecar.SecurityContext)
	}
	out := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(out); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if diff := cmp.Diff(hub.Spec.PodTemplate, out.Spec.PodTemplate); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDCD_IntermediateSpokeDeletesGeneratedFrontendSidecarDropsHubOnlyContainer(t *testing.T) {
	alpha := &DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "stale-generated-sidecar", Namespace: "ns"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				ServiceName: "stale-generated-sidecar",
				FrontendSidecar: &FrontendSidecarSpec{
					Image: "frontend:v1",
				},
			},
		},
	}

	hub := &v1beta1.DynamoComponentDeployment{}
	if err := alpha.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo alpha: %v", err)
	}
	addGeneratedFrontendSidecarHubOnlySecurityContext(t, hub.Spec.PodTemplate)

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom hub: %v", err)
	}
	raw, ok := spoke.Annotations[annDCDSpec]
	if !ok {
		t.Fatalf("expected sparse hub preservation in %q, got %v", annDCDSpec, spoke.Annotations)
	}
	preserved, ok := restoreDCDHubSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDCDSpec, raw)
	}
	if _, ok := findContainerByName(preserved.PodTemplate.Spec.Containers, defaultFrontendSidecarContainerName); !ok {
		t.Fatalf("expected preserved generated frontend sidecar remainder, got %#v", preserved.PodTemplate)
	}

	// Stage 2 edit: the v1alpha1 object removes the generated frontend sidecar
	// field but leaves preservation annotations untouched.
	spoke.Spec.FrontendSidecar = nil

	restoredHub := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo spoke: %v", err)
	}
	if restoredHub.Spec.FrontendSidecar != nil {
		t.Fatalf("stale frontendSidecar reference was restored: %q", *restoredHub.Spec.FrontendSidecar)
	}
	if restoredHub.Spec.PodTemplate != nil {
		if _, ok := findContainerByName(restoredHub.Spec.PodTemplate.Spec.Containers, defaultFrontendSidecarContainerName); ok {
			t.Fatalf("stale generated sidecar container was restored: %#v", restoredHub.Spec.PodTemplate.Spec.Containers)
		}
	}
}

func TestDCD_IntermediateSpokeDeletesFrontendSidecarContainerDropsHubReference(t *testing.T) {
	sidecarName := "sidecar"
	src := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "stale-sidecar", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName:   "stale-sidecar",
				ComponentType:   v1beta1.ComponentTypeFrontend,
				FrontendSidecar: ptr.To(sidecarName),
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "main", Image: "main:v1"},
							{Name: sidecarName, Image: "frontend:v1"},
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
	if spoke.Spec.ExtraPodSpec == nil || spoke.Spec.ExtraPodSpec.PodSpec == nil {
		t.Fatalf("expected sidecar container to be represented in ExtraPodSpec, got %#v", spoke.Spec.ExtraPodSpec)
	}
	if _, ok := findContainerByName(spoke.Spec.ExtraPodSpec.PodSpec.Containers, sidecarName); !ok {
		t.Fatalf("expected sidecar container in spoke ExtraPodSpec, got %#v", spoke.Spec.ExtraPodSpec.PodSpec.Containers)
	}
	raw, ok := spoke.Annotations[annDCDSpec]
	if !ok {
		t.Fatalf("expected sparse hub preservation in %q, got %v", annDCDSpec, spoke.Annotations)
	}
	preserved, ok := restoreDCDHubSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDCDSpec, raw)
	}
	preservedSidecar, ok := findContainerByName(preserved.PodTemplate.Spec.Containers, sidecarName)
	if !ok {
		t.Fatalf("expected preserved podTemplate to carry sidecar key, got %#v", preserved.PodTemplate)
	}
	if preservedSidecar.Image != "" {
		t.Fatalf("expected sparse preserved sidecar key only, got %#v", preservedSidecar)
	}

	// Stage 2 edit: the v1alpha1 object removes the representable sidecar
	// container but leaves preservation annotations untouched.
	spoke.Spec.ExtraPodSpec.PodSpec.Containers = nil

	restoredHub := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if restoredHub.Spec.FrontendSidecar != nil {
		t.Fatalf("stale frontendSidecar reference was restored: %q", *restoredHub.Spec.FrontendSidecar)
	}
	if restoredHub.Spec.PodTemplate != nil {
		if _, ok := findContainerByName(restoredHub.Spec.PodTemplate.Spec.Containers, sidecarName); ok {
			t.Fatalf("stale sidecar container was restored: %#v", restoredHub.Spec.PodTemplate.Spec.Containers)
		}
	}
}

func TestBugDGD_IntermediateSpokeDeletesGeneratedFrontendSidecarDropsHubOnlyContainer(t *testing.T) {
	alpha := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "stale-generated-sidecar", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ServiceName: "frontend",
					FrontendSidecar: &FrontendSidecarSpec{
						Image: "frontend:v1",
					},
				},
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeployment{}
	if err := alpha.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo alpha: %v", err)
	}
	addGeneratedFrontendSidecarHubOnlySecurityContext(t, hub.Spec.Components[0].PodTemplate)

	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom hub: %v", err)
	}
	raw, ok := spoke.Annotations[annDGDSpec]
	if !ok {
		t.Fatalf("expected sparse hub preservation in %q, got %v", annDGDSpec, spoke.Annotations)
	}
	preserved, ok := restoreDGDHubSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDGDSpec, raw)
	}
	if _, ok := findContainerByName(preserved.Components[0].PodTemplate.Spec.Containers, defaultFrontendSidecarContainerName); !ok {
		t.Fatalf("expected preserved generated frontend sidecar remainder, got %#v", preserved.Components[0].PodTemplate)
	}

	// Stage 2 edit: the v1alpha1 object removes the generated frontend sidecar
	// field but leaves preservation annotations untouched.
	component := spoke.Spec.Services["frontend"]
	component.FrontendSidecar = nil

	restoredHub := &v1beta1.DynamoGraphDeployment{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo spoke: %v", err)
	}
	restoredComponent := restoredHub.Spec.Components[0]
	if restoredComponent.FrontendSidecar != nil {
		t.Fatalf("stale frontendSidecar reference was restored: %q", *restoredComponent.FrontendSidecar)
	}
	if restoredComponent.PodTemplate != nil {
		if _, ok := findContainerByName(restoredComponent.PodTemplate.Spec.Containers, defaultFrontendSidecarContainerName); ok {
			t.Fatalf("stale generated sidecar container was restored: %#v", restoredComponent.PodTemplate.Spec.Containers)
		}
	}
}

func TestDGD_IntermediateSpokeDeletesFrontendSidecarContainerDropsHubReference(t *testing.T) {
	sidecarName := "sidecar"
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "stale-sidecar", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:   "frontend",
					ComponentType:   v1beta1.ComponentTypeFrontend,
					FrontendSidecar: ptr.To(sidecarName),
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{Name: "main", Image: "main:v1"},
								{Name: sidecarName, Image: "frontend:v1"},
							},
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
	component := spoke.Spec.Services["frontend"]
	if component == nil || component.ExtraPodSpec == nil || component.ExtraPodSpec.PodSpec == nil {
		t.Fatalf("expected sidecar container to be represented in ExtraPodSpec, got %#v", component)
	}
	if _, ok := findContainerByName(component.ExtraPodSpec.PodSpec.Containers, sidecarName); !ok {
		t.Fatalf("expected sidecar container in spoke ExtraPodSpec, got %#v", component.ExtraPodSpec.PodSpec.Containers)
	}
	raw, ok := spoke.Annotations[annDGDSpec]
	if !ok {
		t.Fatalf("expected sparse hub preservation in %q, got %v", annDGDSpec, spoke.Annotations)
	}
	preserved, ok := restoreDGDHubSpec(raw)
	if !ok {
		t.Fatalf("failed to restore %q payload: %s", annDGDSpec, raw)
	}
	if len(preserved.Components) != 1 {
		t.Fatalf("expected one preserved component, got %#v", preserved.Components)
	}
	preservedSidecar, ok := findContainerByName(preserved.Components[0].PodTemplate.Spec.Containers, sidecarName)
	if !ok {
		t.Fatalf("expected preserved podTemplate to carry sidecar key, got %#v", preserved.Components[0].PodTemplate)
	}
	if preservedSidecar.Image != "" {
		t.Fatalf("expected sparse preserved sidecar key only, got %#v", preservedSidecar)
	}

	// Stage 2 edit: the v1alpha1 object removes the representable sidecar
	// container but leaves preservation annotations untouched.
	component.ExtraPodSpec.PodSpec.Containers = nil

	restoredHub := &v1beta1.DynamoGraphDeployment{}
	if err := spoke.ConvertTo(restoredHub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if len(restoredHub.Spec.Components) != 1 {
		t.Fatalf("expected one restored component, got %#v", restoredHub.Spec.Components)
	}
	restoredComponent := restoredHub.Spec.Components[0]
	if restoredComponent.FrontendSidecar != nil {
		t.Fatalf("stale frontendSidecar reference was restored: %q", *restoredComponent.FrontendSidecar)
	}
	if restoredComponent.PodTemplate != nil {
		if _, ok := findContainerByName(restoredComponent.PodTemplate.Spec.Containers, sidecarName); ok {
			t.Fatalf("stale sidecar container was restored: %#v", restoredComponent.PodTemplate.Spec.Containers)
		}
	}
}
