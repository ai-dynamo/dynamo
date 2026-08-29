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
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const dgdrOutputCopierContainerName = "output-copier"

func TestBugDGDRStaleHubDeployedPhaseRequiresDGDNameMatch(t *testing.T) {
	const newDGDName = "new-deployed-dgd"

	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRStatus: mustDGDRHubStatusAnnotation(t, v1beta1.DynamoGraphDeploymentRequestStatus{
					Phase:   v1beta1.DGDRPhaseDeployed,
					DGDName: "old-dgd",
				}),
			},
		},
		Status: DynamoGraphDeploymentRequestStatus{
			State: DGDRStateReady,
			Deployment: &DeploymentStatus{
				Name: newDGDName,
			},
		},
	}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	if dst.Status.Phase != v1beta1.DGDRPhaseReady {
		t.Fatalf("phase = %q, want %q", dst.Status.Phase, v1beta1.DGDRPhaseReady)
	}
	if dst.Status.DGDName != newDGDName {
		t.Fatalf("dgdName = %q, want %q", dst.Status.DGDName, newDGDName)
	}
}

func TestBugDGDRStaleHubProfilingSubstatusRequiresProfilingPhase(t *testing.T) {
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRStatus: mustDGDRHubStatusAnnotation(t, v1beta1.DynamoGraphDeploymentRequestStatus{
					ProfilingPhase:   v1beta1.ProfilingPhaseSweepingDecode,
					ProfilingJobName: "old-profiling-job",
				}),
			},
		},
		Status: DynamoGraphDeploymentRequestStatus{
			State: DGDRStateReady,
		},
	}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	if dst.Status.Phase != v1beta1.DGDRPhaseReady {
		t.Fatalf("phase = %q, want %q", dst.Status.Phase, v1beta1.DGDRPhaseReady)
	}
	if dst.Status.ProfilingPhase != "" {
		t.Fatalf("profilingPhase = %q, want empty", dst.Status.ProfilingPhase)
	}
	if dst.Status.ProfilingJobName != "" {
		t.Fatalf("profilingJobName = %q, want empty", dst.Status.ProfilingJobName)
	}
}

func TestBugDGDRStaleHubDeploymentInfoRequiresDGDNameMatch(t *testing.T) {
	const newDGDName = "new-info-dgd"

	replicas := int32(3)
	availableReplicas := int32(2)
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRStatus: mustDGDRHubStatusAnnotation(t, v1beta1.DynamoGraphDeploymentRequestStatus{
					DGDName: "old-dgd",
					DeploymentInfo: &v1beta1.DeploymentInfoStatus{
						Replicas:          &replicas,
						AvailableReplicas: &availableReplicas,
					},
				}),
			},
		},
		Status: DynamoGraphDeploymentRequestStatus{
			State: DGDRStateReady,
			Deployment: &DeploymentStatus{
				Name: newDGDName,
			},
		},
	}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	if dst.Status.DGDName != newDGDName {
		t.Fatalf("dgdName = %q, want %q", dst.Status.DGDName, newDGDName)
	}
	if dst.Status.DeploymentInfo != nil {
		t.Fatalf("deploymentInfo = %#v, want nil", dst.Status.DeploymentInfo)
	}
}

func TestBugDGDRStaleAlphaDeploymentDeletedRequiresDGDNameMatch(t *testing.T) {
	const newDGDName = "new-deleted-dgd"

	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRStatus: mustDGDRAlphaStatusAnnotation(t, DynamoGraphDeploymentRequestStatus{
					State: DGDRStateDeploymentDeleted,
					Deployment: &DeploymentStatus{
						Name:    "old-dgd",
						Created: true,
					},
				}),
			},
		},
		Status: v1beta1.DynamoGraphDeploymentRequestStatus{
			Phase:   v1beta1.DGDRPhaseReady,
			DGDName: newDGDName,
		},
	}

	dst := &DynamoGraphDeploymentRequest{}
	if err := dst.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	if dst.Status.State != DGDRStateReady {
		t.Fatalf("state = %q, want %q", dst.Status.State, DGDRStateReady)
	}
	if dst.Status.Deployment == nil {
		t.Fatal("deployment = nil, want minimal live deployment")
	}
	if dst.Status.Deployment.Name != newDGDName {
		t.Fatalf("deployment.name = %q, want %q", dst.Status.Deployment.Name, newDGDName)
	}
	if dst.Status.Deployment.Created {
		t.Fatal("deployment.created = true, want false")
	}
}

func TestBugDGDREmptyAlphaDeploymentStatusRoundTrips(t *testing.T) {
	original := &DynamoGraphDeploymentRequest{
		Status: DynamoGraphDeploymentRequestStatus{
			State:      DGDRStateReady,
			Deployment: &DeploymentStatus{},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := original.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	if diff := cmp.Diff(original.Status, restored.Status); diff != "" {
		t.Fatalf("status mismatch after round-trip (-want +got):\n%s", diff)
	}
}

func TestBugDGDRProfilingConfigPreservesUnprojectableTypedKeys(t *testing.T) {
	config := []byte(`{
		"sla": {
			"ttft": "not-a-number",
			"itl": false,
			"optimizationType": "priority",
			"isl": "1024",
			"osl": null
		},
		"deployment": {
			"modelCache": {
				"pvcName": 123,
				"modelPathInPvc": "",
				"pvcMountPath": false
			}
		},
		"planner": []
	}`)
	src := &DynamoGraphDeploymentRequest{
		Spec: DynamoGraphDeploymentRequestSpec{
			ProfilingConfig: ProfilingConfigSpec{
				Config: &apiextensionsv1.JSON{Raw: config},
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	if restored.Spec.ProfilingConfig.Config == nil {
		t.Fatal("profilingConfig.config = nil, want preserved opaque JSON")
	}
	assertDGDRJSONEqual(t, config, restored.Spec.ProfilingConfig.Config.Raw)
}

func TestBugDGDRProfilingJobResourcesFollowContainerName(t *testing.T) {
	sidecarResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("250m")},
	}
	profilerResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1")},
	}
	updatedProfilerResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("2")},
	}
	tests := []struct {
		name               string
		containers         []corev1.Container
		wantSpokeResources *corev1.ResourceRequirements
		wantRestored       []corev1.Container
	}{
		{
			name: "sidecar only",
			containers: []corev1.Container{{
				Name:      dgdrOutputCopierContainerName,
				Resources: sidecarResources,
			}},
			wantRestored: []corev1.Container{
				{Name: dgdrOutputCopierContainerName, Resources: sidecarResources},
				{Name: dgdrProfilerContainerName, Resources: updatedProfilerResources},
			},
		},
		{
			name: "sidecar before profiler",
			containers: []corev1.Container{
				{Name: dgdrOutputCopierContainerName, Resources: sidecarResources},
				{Name: dgdrProfilerContainerName, Resources: profilerResources},
			},
			wantSpokeResources: &profilerResources,
			wantRestored: []corev1.Container{
				{Name: dgdrOutputCopierContainerName, Resources: sidecarResources},
				{Name: dgdrProfilerContainerName, Resources: updatedProfilerResources},
			},
		},
		{
			name: "profiler before sidecar",
			containers: []corev1.Container{
				{Name: dgdrProfilerContainerName, Resources: profilerResources},
				{Name: dgdrOutputCopierContainerName, Resources: sidecarResources},
			},
			wantSpokeResources: &profilerResources,
			wantRestored: []corev1.Container{
				{Name: dgdrProfilerContainerName, Resources: updatedProfilerResources},
				{Name: dgdrOutputCopierContainerName, Resources: sidecarResources},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Convert named v1beta1 profiling overrides to v1alpha1")
			hub := &v1beta1.DynamoGraphDeploymentRequest{
				Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
					Overrides: &v1beta1.OverridesSpec{
						ProfilingJob: &batchv1.JobSpec{
							Template: corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{Containers: test.containers},
							},
						},
					},
				},
			}
			spoke := &DynamoGraphDeploymentRequest{}
			if err := spoke.ConvertFrom(hub); err != nil {
				t.Fatalf("ConvertFrom() error = %v", err)
			}

			t.Log("Verify only profiler resources are projected into the legacy field")
			if diff := cmp.Diff(test.wantSpokeResources, spoke.Spec.ProfilingConfig.Resources); diff != "" {
				t.Fatalf("profilingConfig.resources mismatch (-want +got):\n%s", diff)
			}

			t.Log("Update legacy profiler resources and convert back to v1beta1")
			spoke.Spec.ProfilingConfig.Resources = &updatedProfilerResources
			restored := &v1beta1.DynamoGraphDeploymentRequest{}
			if err := spoke.ConvertTo(restored); err != nil {
				t.Fatalf("ConvertTo() error = %v", err)
			}

			t.Log("Verify resources remain attached to their named containers")
			if restored.Spec.Overrides == nil || restored.Spec.Overrides.ProfilingJob == nil {
				t.Fatal("profilingJob override = nil, want restored named containers")
			}
			got := restored.Spec.Overrides.ProfilingJob.Template.Spec.Containers
			if diff := cmp.Diff(test.wantRestored, got); diff != "" {
				t.Fatalf("profiling containers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestBugDGDRProfilerFirstHubRoundTripPreservesOrder(t *testing.T) {
	profilerResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1")},
	}
	sidecarResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("250m")},
	}
	wantContainers := []corev1.Container{
		{Name: dgdrProfilerContainerName, Resources: profilerResources},
		{Name: dgdrOutputCopierContainerName, Resources: sidecarResources},
	}

	t.Log("Start from a profiler-first v1beta1 override list")
	hub := &v1beta1.DynamoGraphDeploymentRequest{
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Overrides: &v1beta1.OverridesSpec{
				ProfilingJob: &batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Containers: wantContainers},
					},
				},
			},
		},
	}

	t.Log("Convert hub → spoke → hub without changing alpha fields")
	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	t.Log("Verify profiler-first ordering and resources survive the round trip")
	if restored.Spec.Overrides == nil || restored.Spec.Overrides.ProfilingJob == nil {
		t.Fatal("profilingJob override = nil, want profiler-first containers")
	}
	got := restored.Spec.Overrides.ProfilingJob.Template.Spec.Containers
	if diff := cmp.Diff(wantContainers, got); diff != "" {
		t.Fatalf("profiling containers mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDGDRAlphaResourcesRoundTripOmitsPrivateAnnotation(t *testing.T) {
	resources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("2")},
	}

	t.Log("Start from alpha-only profilingConfig.resources")
	spoke := &DynamoGraphDeploymentRequest{
		Spec: DynamoGraphDeploymentRequestSpec{
			ProfilingConfig: ProfilingConfigSpec{
				Resources: &resources,
			},
		},
	}

	t.Log("Convert spoke → hub → spoke")
	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if _, ok := hub.Annotations[annDGDRSpec]; ok {
		t.Fatalf("hub annotations[%q] = %q, want absent for alpha-only resources", annDGDRSpec, hub.Annotations[annDGDRSpec])
	}
	if hub.Spec.Overrides == nil || hub.Spec.Overrides.ProfilingJob == nil {
		t.Fatal("hub profilingJob = nil, want named profiler projection")
	}
	gotHubContainers := hub.Spec.Overrides.ProfilingJob.Template.Spec.Containers
	wantHubContainers := []corev1.Container{{
		Name:      dgdrProfilerContainerName,
		Resources: resources,
	}}
	if diff := cmp.Diff(wantHubContainers, gotHubContainers); diff != "" {
		t.Fatalf("hub profiling containers mismatch (-want +got):\n%s", diff)
	}

	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	t.Log("Verify resources round-trip without introducing nvidia.com/dgdr-spec")
	if diff := cmp.Diff(&resources, restored.Spec.ProfilingConfig.Resources); diff != "" {
		t.Fatalf("profilingConfig.resources mismatch (-want +got):\n%s", diff)
	}
	if _, ok := restored.Annotations[annDGDRSpec]; ok {
		t.Fatalf("spoke annotations[%q] present after round trip, want absent", annDGDRSpec)
	}
}

func TestBugDGDRSidecarFirstHubRoundTripPreservesOrder(t *testing.T) {
	profilerResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1")},
	}
	sidecarResources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("250m")},
	}
	wantContainers := []corev1.Container{
		{Name: dgdrOutputCopierContainerName, Resources: sidecarResources},
		{Name: dgdrProfilerContainerName, Resources: profilerResources},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Overrides: &v1beta1.OverridesSpec{
				ProfilingJob: &batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Containers: wantContainers},
					},
				},
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	got := restored.Spec.Overrides.ProfilingJob.Template.Spec.Containers
	if diff := cmp.Diff(wantContainers, got); diff != "" {
		t.Fatalf("profiling containers mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDGDRNameOnlyProfilerPreserved(t *testing.T) {
	wantContainers := []corev1.Container{{Name: dgdrProfilerContainerName}}

	hub := &v1beta1.DynamoGraphDeploymentRequest{
		Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
			Overrides: &v1beta1.OverridesSpec{
				ProfilingJob: &batchv1.JobSpec{
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Containers: wantContainers},
					},
				},
			},
		},
	}

	spoke := &DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if spoke.Spec.ProfilingConfig.Resources != nil {
		t.Fatalf("profilingConfig.resources = %#v, want nil for name-only profiler", spoke.Spec.ProfilingConfig.Resources)
	}
	restored := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	got := restored.Spec.Overrides.ProfilingJob.Template.Spec.Containers
	if diff := cmp.Diff(wantContainers, got); diff != "" {
		t.Fatalf("profiling containers mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDGDRLegacyUnnamedAnnotationMigratesInPlace(t *testing.T) {
	resources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("2")},
	}
	sidecarImage := "custom-output-copier:1"

	t.Log("Seed a spoke object with a legacy unnamed profiler annotation")
	spoke := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRSpec: mustDGDRHubSpecAnnotation(t, v1beta1.DynamoGraphDeploymentRequestSpec{
					Overrides: &v1beta1.OverridesSpec{
						ProfilingJob: &batchv1.JobSpec{
							Template: corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{Name: "", Image: "custom-profiler:1"},
										{Name: dgdrOutputCopierContainerName, Image: sidecarImage},
									},
								},
							},
						},
					},
				}),
			},
		},
		Spec: DynamoGraphDeploymentRequestSpec{
			ProfilingConfig: ProfilingConfigSpec{
				Resources: &resources,
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	want := []corev1.Container{
		{Name: dgdrProfilerContainerName, Image: "custom-profiler:1", Resources: resources},
		{Name: dgdrOutputCopierContainerName, Image: sidecarImage},
	}
	got := hub.Spec.Overrides.ProfilingJob.Template.Spec.Containers
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("migrated profiling containers mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDGDRLiveHubUnnamedProfilerMigratesOnRoundTrip(t *testing.T) {
	resources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("2")},
	}
	sidecarImage := "custom-output-copier:1"
	tests := []struct {
		name         string
		containers   []corev1.Container
		wantRestored []corev1.Container
	}{
		{
			name: "sole unnamed profiler",
			containers: []corev1.Container{
				{Name: "", Resources: resources},
			},
			wantRestored: []corev1.Container{
				{Name: dgdrProfilerContainerName, Resources: resources},
			},
		},
		{
			name: "unnamed profiler before sidecar",
			containers: []corev1.Container{
				{Name: "", Resources: resources},
				{Name: dgdrOutputCopierContainerName, Image: sidecarImage},
			},
			wantRestored: []corev1.Container{
				{Name: dgdrProfilerContainerName, Resources: resources},
				{Name: dgdrOutputCopierContainerName, Image: sidecarImage},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Start from a stored hub object with a leftover unnamed profiler")
			hub := &v1beta1.DynamoGraphDeploymentRequest{
				Spec: v1beta1.DynamoGraphDeploymentRequestSpec{
					Overrides: &v1beta1.OverridesSpec{
						ProfilingJob: &batchv1.JobSpec{
							Template: corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{Containers: test.containers},
							},
						},
					},
				},
			}

			t.Log("Convert hub → spoke without mutating the stored hub containers")
			spoke := &DynamoGraphDeploymentRequest{}
			if err := spoke.ConvertFrom(hub); err != nil {
				t.Fatalf("ConvertFrom() error = %v", err)
			}
			if hub.Spec.Overrides.ProfilingJob.Template.Spec.Containers[0].Name != "" {
				t.Fatalf("stored hub container name = %q, want leftover empty name", hub.Spec.Overrides.ProfilingJob.Template.Spec.Containers[0].Name)
			}
			if diff := cmp.Diff(&resources, spoke.Spec.ProfilingConfig.Resources); diff != "" {
				t.Fatalf("profilingConfig.resources mismatch (-want +got):\n%s", diff)
			}

			t.Log("Convert spoke → hub and rewrite the leftover entry as name:profiler")
			restored := &v1beta1.DynamoGraphDeploymentRequest{}
			if err := spoke.ConvertTo(restored); err != nil {
				t.Fatalf("ConvertTo() error = %v", err)
			}
			got := restored.Spec.Overrides.ProfilingJob.Template.Spec.Containers
			if diff := cmp.Diff(test.wantRestored, got); diff != "" {
				t.Fatalf("profiling containers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestBugDGDRLegacyUnnamedKeptDistinctWhenProfilerExists(t *testing.T) {
	spoke := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRSpec: mustDGDRHubSpecAnnotation(t, v1beta1.DynamoGraphDeploymentRequestSpec{
					Overrides: &v1beta1.OverridesSpec{
						ProfilingJob: &batchv1.JobSpec{
							Template: corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{
									Containers: []corev1.Container{
										{Name: dgdrProfilerContainerName, Image: "named-profiler:1"},
										{Name: "", Image: "unnamed-extra:1"},
										{Name: dgdrOutputCopierContainerName, Image: "sidecar:1"},
									},
								},
							},
						},
					},
				}),
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := spoke.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	want := []corev1.Container{
		{Name: dgdrProfilerContainerName, Image: "named-profiler:1"},
		{Name: "", Image: "unnamed-extra:1"},
		{Name: dgdrOutputCopierContainerName, Image: "sidecar:1"},
	}
	got := hub.Spec.Overrides.ProfilingJob.Template.Spec.Containers
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("profiling containers mismatch (-want +got):\n%s", diff)
	}
}

func mustDGDRHubSpecAnnotation(t *testing.T, spec v1beta1.DynamoGraphDeploymentRequestSpec) string {
	t.Helper()
	data, err := marshalDGDRHubSpec(&spec)
	if err != nil {
		t.Fatalf("marshal DGDR hub spec annotation: %v", err)
	}
	return string(data)
}

func mustDGDRHubStatusAnnotation(t *testing.T, status v1beta1.DynamoGraphDeploymentRequestStatus) string {
	t.Helper()
	data, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("marshal DGDR hub status annotation: %v", err)
	}
	return string(data)
}

func assertDGDRJSONEqual(t *testing.T, wantRaw, gotRaw []byte) {
	t.Helper()
	var want, got any
	if err := json.Unmarshal(wantRaw, &want); err != nil {
		t.Fatalf("unmarshal wanted JSON: %v", err)
	}
	if err := json.Unmarshal(gotRaw, &got); err != nil {
		t.Fatalf("unmarshal got JSON: %v", err)
	}
	if reflect.DeepEqual(want, got) {
		return
	}
	wantJSON, _ := json.Marshal(want)
	gotJSON, _ := json.Marshal(got)
	t.Fatalf("JSON mismatch:\nwant: %s\n got: %s", wantJSON, gotJSON)
}

func mustDGDRAlphaStatusAnnotation(t *testing.T, status DynamoGraphDeploymentRequestStatus) string {
	t.Helper()
	data, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("marshal DGDR alpha status annotation: %v", err)
	}
	return string(data)
}
