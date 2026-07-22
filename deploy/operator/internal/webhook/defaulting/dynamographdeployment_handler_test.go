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

package defaulting

import (
	"context"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// admissionCtx builds a context carrying an admission request for the given operation and kind.
func admissionCtx(op admissionv1.Operation, kind schema.GroupVersionKind) context.Context {
	return admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: op,
			Kind: metav1.GroupVersionKind{
				Group:   kind.Group,
				Version: kind.Version,
				Kind:    kind.Kind,
			},
		},
	})
}

func TestDGDDefaulter_Default(t *testing.T) {
	const testVersion = "0.8.0"

	tests := []struct {
		name            string
		operatorVersion string
		ctx             context.Context
		dgd             *nvidiacomv1beta1.DynamoGraphDeployment
		wantAnnotation  string
		wantErr         bool
	}{
		{
			name:            "CREATE stamps operator version on new DGD without annotations",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Create, nvidiacomv1beta1.DynamoGraphDeploymentGVK),
			dgd: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: testVersion,
		},
		{
			name:            "CREATE stamps operator version on DGD with existing annotations",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Create, nvidiacomv1beta1.DynamoGraphDeploymentGVK),
			dgd: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					Annotations: map[string]string{
						"some-other-annotation": "some-value",
					},
				},
			},
			wantAnnotation: testVersion,
		},
		{
			name:            "CREATE does not overwrite pre-existing origin version",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Create, nvidiacomv1beta1.DynamoGraphDeploymentGVK),
			dgd: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "0.7.0",
					},
				},
			},
			wantAnnotation: "0.7.0",
		},
		{
			name:            "UPDATE does not stamp annotation",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Update, nvidiacomv1beta1.DynamoGraphDeploymentGVK),
			dgd: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: "",
		},
		{
			name:            "UPDATE preserves existing annotation",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Update, nvidiacomv1beta1.DynamoGraphDeploymentGVK),
			dgd: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "0.7.0",
					},
				},
			},
			wantAnnotation: "0.7.0",
		},
		{
			name:            "DELETE does not stamp annotation",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Delete, nvidiacomv1beta1.DynamoGraphDeploymentGVK),
			dgd: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: "",
		},
		{
			name:            "no admission request in context fails closed",
			operatorVersion: testVersion,
			ctx:             context.Background(),
			dgd: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: "",
			wantErr:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defaulter := NewDGDDefaulter(tt.operatorVersion, false)

			err := defaulter.Default(tt.ctx, tt.dgd)
			if (err != nil) != tt.wantErr {
				t.Errorf("Default() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			got := ""
			if tt.dgd.Annotations != nil {
				got = tt.dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]
			}

			if got != tt.wantAnnotation {
				t.Errorf("annotation %q = %q, want %q",
					consts.KubeAnnotationDynamoOperatorOriginVersion, got, tt.wantAnnotation)
			}
		})
	}
}

func TestDGDDefaulter_DefaultsNilReplicas(t *testing.T) {
	tests := []struct {
		name         string
		op           admissionv1.Operation
		components   []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		wantReplicas map[string]int32
	}{
		{
			name: "CREATE defaults nil replicas to 1",
			op:   admissionv1.Create,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Frontend", Replicas: nil},
				{ComponentName: "VllmWorker", Replicas: ptr.To(int32(3))},
				{ComponentName: "NewComponent", Replicas: nil},
			},
			wantReplicas: map[string]int32{
				"Frontend":     1,
				"VllmWorker":   3,
				"NewComponent": 1,
			},
		},
		{
			name: "UPDATE defaults nil replicas to 1",
			op:   admissionv1.Update,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "NewComponent", Replicas: nil},
			},
			wantReplicas: map[string]int32{
				"NewComponent": 1,
			},
		},
		{
			name: "does not overwrite explicit replicas",
			op:   admissionv1.Create,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(5))},
			},
			wantReplicas: map[string]int32{
				"Worker": 5,
			},
		},
		{
			name: "preserves explicit zero replicas",
			op:   admissionv1.Create,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Idle", Replicas: ptr.To(int32(0))},
			},
			wantReplicas: map[string]int32{
				"Idle": 0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defaulter := NewDGDDefaulter("0.9.0", false)
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: tt.components,
				},
			}

			if err := defaulter.Default(admissionCtx(tt.op, nvidiacomv1beta1.DynamoGraphDeploymentGVK), dgd); err != nil {
				t.Fatalf("Default() unexpected error: %v", err)
			}

			for name, want := range tt.wantReplicas {
				component := dgd.GetComponentByName(name)
				if component == nil {
					t.Fatalf("component %q not found", name)
				}
				if component.Replicas == nil {
					t.Errorf("component %q: replicas is nil, want %d", name, want)
					continue
				}
				if *component.Replicas != want {
					t.Errorf("component %q: replicas = %d, want %d", name, *component.Replicas, want)
				}
			}
		})
	}
}

func TestDGDDefaulter_DefaultsGroveMinAvailable(t *testing.T) {
	tests := []struct {
		name             string
		op               admissionv1.Operation
		groveEnabled     bool
		annotations      map[string]string
		components       []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		wantMinAvailable map[string]*int32
	}{
		{
			name:         "CREATE defaults nil replicas to minAvailable 1 on Grove pathway",
			op:           admissionv1.Create,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: nil},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": ptr.To(int32(1)),
			},
		},
		{
			name:         "UPDATE defaults positive replicas to minAvailable 1 on Grove pathway",
			op:           admissionv1.Update,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(3))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": ptr.To(int32(1)),
			},
		},
		{
			name:         "defaults zero replicas to minAvailable 1 on Grove pathway",
			op:           admissionv1.Create,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Idle", Replicas: ptr.To(int32(0))},
			},
			wantMinAvailable: map[string]*int32{
				"Idle": ptr.To(int32(1)),
			},
		},
		{
			name:         "preserves explicit minAvailable",
			op:           admissionv1.Create,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(3)), MinAvailable: ptr.To(int32(2))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": ptr.To(int32(2)),
			},
		},
		{
			name:         "CREATE preserves explicit zero minAvailable for validation",
			op:           admissionv1.Create,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(1)), MinAvailable: ptr.To(int32(0))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": ptr.To(int32(0)),
			},
		},
		{
			name:         "UPDATE preserves minAvailable when replicas become positive",
			op:           admissionv1.Update,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(3)), MinAvailable: ptr.To(int32(2))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": ptr.To(int32(2)),
			},
		},
		{
			name:         "UPDATE preserves minAvailable when replicas become zero",
			op:           admissionv1.Update,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(0)), MinAvailable: ptr.To(int32(1))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": ptr.To(int32(1)),
			},
		},
		{
			name:         "UPDATE preserves explicit minAvailable away from zero boundary",
			op:           admissionv1.Update,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(4)), MinAvailable: ptr.To(int32(2))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": ptr.To(int32(2)),
			},
		},
		{
			name:         "UPDATE preserves explicit zero minAvailable for validation",
			op:           admissionv1.Update,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(1)), MinAvailable: ptr.To(int32(0))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": ptr.To(int32(0)),
			},
		},
		{
			name:         "does not default minAvailable when operator disables Grove",
			op:           admissionv1.Create,
			groveEnabled: false,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(3))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": nil,
			},
		},
		{
			name:         "does not default minAvailable when DGD opts out of Grove",
			op:           admissionv1.Create,
			groveEnabled: true,
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: "false",
			},
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(3))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": nil,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defaulter := NewDGDDefaulter("0.9.0", tt.groveEnabled)
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test",
					Namespace:   "default",
					Annotations: tt.annotations,
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: tt.components,
				},
			}
			ctx := admissionCtx(tt.op, nvidiacomv1beta1.DynamoGraphDeploymentGVK)

			if err := defaulter.Default(ctx, dgd); err != nil {
				t.Fatalf("Default() unexpected error: %v", err)
			}

			for name, want := range tt.wantMinAvailable {
				component := dgd.GetComponentByName(name)
				if component == nil {
					t.Fatalf("component %q not found", name)
				}
				if want == nil {
					if component.MinAvailable != nil {
						t.Errorf("component %q: minAvailable = %d, want nil", name, *component.MinAvailable)
					}
					continue
				}
				if component.MinAvailable == nil {
					t.Errorf("component %q: minAvailable is nil, want %d", name, *want)
					continue
				}
				if *component.MinAvailable != *want {
					t.Errorf("component %q: minAvailable = %d, want %d", name, *component.MinAvailable, *want)
				}
			}
		})
	}
}

func TestDGDV1Alpha1Defaulter_Default(t *testing.T) {
	dgd := &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
				},
			},
		},
	}
	defaulter := &dgdV1Alpha1Defaulter{defaulter: NewDGDDefaulter("0.9.0", false)}

	if err := defaulter.Default(admissionCtx(admissionv1.Create, nvidiacomv1alpha1.DynamoGraphDeploymentGVK), dgd); err != nil {
		t.Fatalf("Default() unexpected error: %v", err)
	}
	if got := dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]; got != "0.9.0" {
		t.Errorf("origin annotation = %q, want %q", got, "0.9.0")
	}
	if got := dgd.Spec.Services["worker"].Replicas; got == nil || *got != 1 {
		t.Errorf("worker replicas = %v, want 1", got)
	}
}
