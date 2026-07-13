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
	"slices"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestDCD_DivergedLegacyCarrierDoesNotOverrideSparseSpokeSave(t *testing.T) {
	savedNamespace := "saved"
	data, err := marshalDCDSpokeSpec(&DynamoComponentDeploymentSpec{
		DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
			DynamoNamespace: &savedNamespace,
		},
	}, false)
	if err != nil {
		t.Fatalf("marshal DCD spoke spec: %v", err)
	}

	hub := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcd",
			Namespace: "ns",
			Annotations: map[string]string{
				annDCDSpec:                        string(data),
				"nvidia.com/dcd-dynamo-namespace": "stale",
			},
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "dcd",
			},
		},
	}

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if spoke.Spec.DynamoNamespace == nil || *spoke.Spec.DynamoNamespace != savedNamespace {
		t.Fatalf("DynamoNamespace = %v, want %q from sparse save", spoke.Spec.DynamoNamespace, savedNamespace)
	}
}

// TestBugDCD_CustomMainNameAlphaEnvEditDoesNotSpawnLiteralMain reproduces a
// corruption of custom-named main containers through v1alpha1 edits: the
// decomposition used to key the semantic main container on the literal "main"
// instead of the live hub mainContainerName, so a hub spec with
// mainContainerName="engine" converted its "engine" container to a plain
// extraPodSpec sidecar. A v1alpha1 client editing the flat Envs field then
// composed those envs into a NEW literal "main" container, producing a hub
// object with both a "main" container and mainContainerName="engine", which
// the validation webhook rejects as ambiguous.
func TestBugDCD_CustomMainNameAlphaEnvEditDoesNotSpawnLiteralMain(t *testing.T) {
	hub := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "custom-main", Namespace: "ns"},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName:             "custom-main",
				MainContainerNameOverride: "engine",
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "engine", Image: "dynamo:latest"},
							{Name: "logger", Image: "fluent/fluent-bit"},
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

	// A v1alpha1 client edits the flat Envs field.
	spoke.Spec.Envs = append(spoke.Spec.Envs, corev1.EnvVar{Name: "ADDED", Value: "1"})

	got := &v1beta1.DynamoComponentDeployment{}
	if err := spoke.ConvertTo(got); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	if got.Spec.MainContainerNameOverride != "engine" {
		t.Fatalf("mainContainerName = %q, want %q", got.Spec.MainContainerNameOverride, "engine")
	}
	if got.Spec.PodTemplate == nil {
		t.Fatal("podTemplate is nil")
	}
	if hasContainerNamed(got.Spec.PodTemplate.Spec.Containers, "main") {
		t.Fatalf("podTemplate grew a literal \"main\" container alongside mainContainerName=%q: %#v",
			got.Spec.MainContainerNameOverride, got.Spec.PodTemplate.Spec.Containers)
	}
	engine, ok := findContainerByName(got.Spec.PodTemplate.Spec.Containers, "engine")
	if !ok {
		t.Fatalf("engine container missing: %#v", got.Spec.PodTemplate.Spec.Containers)
	}
	if engine.Image != "dynamo:latest" {
		t.Fatalf("engine image = %q, want %q", engine.Image, "dynamo:latest")
	}
	if !slices.ContainsFunc(engine.Env, func(env corev1.EnvVar) bool {
		return env.Name == "ADDED" && env.Value == "1"
	}) {
		t.Fatalf("engine container is missing the edited flat env, got %#v", engine.Env)
	}
	logger, ok := findContainerByName(got.Spec.PodTemplate.Spec.Containers, "logger")
	if !ok || logger.Image != "fluent/fluent-bit" {
		t.Fatalf("logger sidecar not intact: %#v", got.Spec.PodTemplate.Spec.Containers)
	}
}
