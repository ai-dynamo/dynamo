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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestBugDGD_HubClearsPVCAnnotationDropsSavedSpokeSpec(t *testing.T) {
	create := true
	name := "old-model-pvc"
	alpha := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc"},
		Spec: DynamoGraphDeploymentSpec{
			PVCs: []PVC{{
				Create:           &create,
				Name:             &name,
				StorageClass:     "standard",
				Size:             resource.MustParse("10Gi"),
				VolumeAccessMode: corev1.ReadWriteOnce,
			}},
		},
	}

	hub := &v1beta1.DynamoGraphDeployment{}
	if err := alpha.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	delete(hub.Annotations, annDGDPVCs)

	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if len(spoke.Spec.PVCs) > 0 {
		t.Fatalf("PVCs restored from stale %q: %#v", annDGDSpokeSpec, spoke.Spec.PVCs)
	}
}

func TestBugDGD_HubClearsComponentAnnotationDropsSavedSpokeSpec(t *testing.T) {
	alpha := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "component-annotation"},
		Spec: DynamoGraphDeploymentSpec{
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
	if err := alpha.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	delete(hub.Annotations, newDGDComponentCarrier(&hub.ObjectMeta, "worker").key(suffixAutoscaling))

	spoke := &DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}
	if got := spoke.Spec.Services["worker"].Autoscaling; got != nil {
		t.Fatalf("autoscaling restored from stale %q: %#v", annDGDSpokeSpec, got)
	}
}
