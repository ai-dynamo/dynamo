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
	"github.com/google/go-cmp/cmp/cmpopts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestBugDGD_IntermediateHubAddsMainOnlyPodTemplateRoundTrips(t *testing.T) {
	alpha := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "main-only-pod-template", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(v1beta1.ComponentTypeWorker),
				},
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeployment{}
	if err := alpha.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	hub.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: mainContainerName}},
		},
	}

	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	restored := &v1beta1.DynamoGraphDeployment{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if diff := cmp.Diff(hub.Spec.Components[0].PodTemplate, restored.Spec.Components[0].PodTemplate, cmpopts.EquateEmpty()); diff != "" {
		t.Fatalf("podTemplate mismatch (-want +got):\n%s", diff)
	}
}

func TestBugDGD_BetaPrefillDecodeConvertFromUsesAlphaSubComponentType(t *testing.T) {
	hub := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "disaggregated", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "prefill",
					ComponentType: v1beta1.ComponentTypePrefill,
				},
				{
					ComponentName: "decode",
					ComponentType: v1beta1.ComponentTypeDecode,
				},
			},
		},
	}

	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	for _, name := range []string{"prefill", "decode"} {
		component := spoke.Spec.Services[name]
		if component == nil {
			t.Fatalf("expected %q service after ConvertFrom: %#v", name, spoke.Spec.Services)
		}
		if component.ComponentType != string(v1beta1.ComponentTypeWorker) {
			t.Fatalf("%s componentType = %q, want worker", name, component.ComponentType)
		}
		if component.SubComponentType != name {
			t.Fatalf("%s subComponentType = %q, want %q", name, component.SubComponentType, name)
		}
	}

	restored := &v1beta1.DynamoGraphDeployment{}
	if err := spoke.ConvertTo(restored); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	if diff := cmp.Diff(hub.Spec.Components, restored.Spec.Components, cmpopts.EquateEmpty()); diff != "" {
		t.Fatalf("components mismatch (-want +got):\n%s", diff)
	}
}
