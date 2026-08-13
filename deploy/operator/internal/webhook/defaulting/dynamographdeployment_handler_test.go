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
	"encoding/json"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// admissionCtx builds a context carrying an admission request for the given operation and kind.
func admissionCtx(op admissionv1.Operation, kind schema.GroupVersionKind) context.Context {
	ctx := admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: op,
			Kind: metav1.GroupVersionKind{
				Group:   kind.Group,
				Version: kind.Version,
				Kind:    kind.Kind,
			},
		},
	})
	return features.WithGate(ctx, features.Defaults())
}

func admissionCtxWithOld(
	t *testing.T,
	op admissionv1.Operation,
	kind schema.GroupVersionKind,
	old runtime.Object,
) context.Context {
	t.Helper()
	raw, err := json.Marshal(old)
	if err != nil {
		t.Fatalf("marshal old admission object: %v", err)
	}
	ctx := admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: op,
			Kind: metav1.GroupVersionKind{
				Group:   kind.Group,
				Version: kind.Version,
				Kind:    kind.Kind,
			},
			OldObject: runtime.RawExtension{Raw: raw},
		},
	})
	return features.WithGate(ctx, features.Defaults())
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
			defaulter := NewDGDDefaulter(tt.operatorVersion)

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
			defaulter := NewDGDDefaulter("0.9.0")
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
		wantProvider     string
		wantUnselected   bool
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
			name:         "CREATE ignores a user-supplied provider and follows routing intent",
			op:           admissionv1.Create,
			groveEnabled: true,
			annotations: map[string]string{
				consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove,
				consts.KubeAnnotationEnableGrove:      consts.KubeLabelValueFalse,
			},
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(3))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": nil,
			},
			wantProvider: consts.WorkloadProviderComponent,
		},
		{
			name:         "legacy UPDATE remains unselected for controller adoption",
			op:           admissionv1.Update,
			groveEnabled: true,
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(3))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": nil,
			},
			wantUnselected: true,
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
		{
			name:         "selected Grove provider remains authoritative when the feature gate is disabled",
			op:           admissionv1.Update,
			groveEnabled: false,
			annotations: map[string]string{
				consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove,
				consts.KubeAnnotationEnableGrove:      consts.KubeLabelValueFalse,
			},
			components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Worker", Replicas: ptr.To(int32(3))},
			},
			wantMinAvailable: map[string]*int32{
				"Worker": ptr.To(int32(1)),
			},
		},
		{
			name:         "selected component provider remains authoritative when Grove is enabled",
			op:           admissionv1.Update,
			groveEnabled: true,
			annotations: map[string]string{
				consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent,
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
			t.Log("Build a DGD and defaulter for the provider-defaulting scenario")
			defaulter := NewDGDDefaulter("0.9.0")
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
			if tt.op == admissionv1.Update {
				ctx = admissionCtxWithOld(t, tt.op, nvidiacomv1beta1.DynamoGraphDeploymentGVK, dgd.DeepCopy())
			}
			ctx = features.WithGate(ctx, features.Gates{Grove: tt.groveEnabled})

			t.Log("Apply level-based component defaults")
			if err := defaulter.Default(ctx, dgd); err != nil {
				t.Fatalf("Default() unexpected error: %v", err)
			}

			t.Log("Verify provider selection and component minimum availability")
			if tt.wantProvider != "" {
				if got := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]; got != tt.wantProvider {
					t.Errorf("workload provider = %q, want %q", got, tt.wantProvider)
				}
			}
			if tt.wantUnselected {
				if _, exists := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]; exists {
					t.Errorf("provider annotation was materialized before controller adoption")
				}
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

func TestDGDDefaulter_GroveWorkerHashSuffix(t *testing.T) {
	tests := []struct {
		name         string
		op           admissionv1.Operation
		groveEnabled bool
		old          func() *nvidiacomv1beta1.DynamoGraphDeployment
		mutate       func(*nvidiacomv1beta1.DynamoGraphDeployment)
		wantSuffix   bool
		wantProvider string
	}{
		{
			name:         "CREATE sets the suffix for a newly selected Grove provider",
			op:           admissionv1.Create,
			groveEnabled: true,
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.AnnotationGroveWorkerHashSuffixEnabled: consts.KubeLabelValueTrue,
				}
			},
			wantSuffix:   true,
			wantProvider: consts.WorkloadProviderGrove,
		},
		{
			name:         "UPDATE preserves an unsuffixed Grove namespace for a frontend-only change",
			op:           admissionv1.Update,
			groveEnabled: true,
			old: func() *nvidiacomv1beta1.DynamoGraphDeployment {
				return groveWorkerHashSuffixTestDGD(consts.WorkloadProviderGrove)
			},
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				frontend := dgd.GetComponentByName("frontend")
				frontend.PodTemplate.Spec.Containers[0].Env = append(frontend.PodTemplate.Spec.Containers[0].Env,
					corev1.EnvVar{Name: "FRONTEND_CONFIG_REVISION", Value: "2"})
			},
			wantProvider: consts.WorkloadProviderGrove,
		},
		{
			name:         "UPDATE uses the selected Grove provider when the feature gate and routing intent change",
			op:           admissionv1.Update,
			groveEnabled: false,
			old: func() *nvidiacomv1beta1.DynamoGraphDeployment {
				return groveWorkerHashSuffixTestDGD(consts.WorkloadProviderGrove)
			},
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations[consts.KubeAnnotationEnableGrove] = consts.KubeLabelValueFalse
				prefill := dgd.GetComponentByName("prefill")
				prefill.PodTemplate.Spec.Containers[0].Env[0].Value = "8192"
			},
			wantSuffix:   true,
			wantProvider: consts.WorkloadProviderGrove,
		},
		{
			name:         "UPDATE normalizes old defaultable worker fields before comparing",
			op:           admissionv1.Update,
			groveEnabled: false,
			old: func() *nvidiacomv1beta1.DynamoGraphDeployment {
				return groveWorkerHashSuffixTestDGD(consts.WorkloadProviderGrove)
			},
			wantProvider: consts.WorkloadProviderGrove,
		},
		{
			name:         "UPDATE preserves a suffix enabled by an earlier Grove admission",
			op:           admissionv1.Update,
			groveEnabled: false,
			old: func() *nvidiacomv1beta1.DynamoGraphDeployment {
				dgd := groveWorkerHashSuffixTestDGD(consts.WorkloadProviderGrove)
				dgd.Annotations[consts.AnnotationGroveWorkerHashSuffixEnabled] = consts.KubeLabelValueTrue
				return dgd
			},
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				delete(dgd.Annotations, consts.AnnotationGroveWorkerHashSuffixEnabled)
			},
			wantSuffix:   true,
			wantProvider: consts.WorkloadProviderGrove,
		},
		{
			name:         "UPDATE removes a user-supplied suffix marker for a component provider",
			op:           admissionv1.Update,
			groveEnabled: true,
			old: func() *nvidiacomv1beta1.DynamoGraphDeployment {
				return groveWorkerHashSuffixTestDGD(consts.WorkloadProviderComponent)
			},
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations[consts.AnnotationGroveWorkerHashSuffixEnabled] = consts.KubeLabelValueTrue
				prefill := dgd.GetComponentByName("prefill")
				prefill.PodTemplate.Spec.Containers[0].Env[0].Value = "8192"
			},
			wantProvider: consts.WorkloadProviderComponent,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build the incoming DGD and preserve the old object for update admission")
			var old *nvidiacomv1beta1.DynamoGraphDeployment
			var dgd *nvidiacomv1beta1.DynamoGraphDeployment
			if tt.old == nil {
				dgd = groveWorkerHashSuffixTestDGD("")
			} else {
				old = tt.old()
				dgd = old.DeepCopy()
			}
			if tt.mutate != nil {
				tt.mutate(dgd)
			}

			t.Log("Run defaulting with the configured feature-gate state")
			ctx := admissionCtx(tt.op, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
			if old != nil {
				ctx = admissionCtxWithOld(t, tt.op, nvidiacomv1beta1.DynamoGraphDeploymentGVK, old)
			}
			ctx = features.WithGate(ctx, features.Gates{Grove: tt.groveEnabled})
			if err := NewDGDDefaulter("0.9.0").Default(ctx, dgd); err != nil {
				t.Fatalf("Default() unexpected error: %v", err)
			}

			t.Log("Verify immutable provider selection and operator-owned suffix state")
			if got := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]; got != tt.wantProvider {
				t.Errorf("workload provider = %q, want %q", got, tt.wantProvider)
			}
			gotSuffix := dgd.Annotations[consts.AnnotationGroveWorkerHashSuffixEnabled] == consts.KubeLabelValueTrue
			if gotSuffix != tt.wantSuffix {
				t.Errorf("worker hash suffix enabled = %t, want %t", gotSuffix, tt.wantSuffix)
			}
		})
	}
}

func TestDGDDefaulter_GroveWorkerHashSuffixRejectsUpdateWithoutOldDGD(t *testing.T) {
	t.Log("Build an immutable Grove DGD without an admission old object")
	dgd := groveWorkerHashSuffixTestDGD(consts.WorkloadProviderGrove)
	ctx := admissionCtx(admissionv1.Update, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
	ctx = features.WithGate(ctx, features.Gates{Grove: true})

	t.Log("Default the malformed update admission")
	err := NewDGDDefaulter("0.9.0").Default(ctx, dgd)

	t.Log("Reject the update because its operator-owned migration state is unknown")
	if err == nil {
		t.Fatal("Default() error = nil, want error for UPDATE without old DynamoGraphDeployment")
	}

}
func groveWorkerHashSuffixTestDGD(provider string) *nvidiacomv1beta1.DynamoGraphDeployment {
	component := func(
		name string,
		componentType nvidiacomv1beta1.ComponentType,
		image string,
	) nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
		return nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: name,
			ComponentType: componentType,
			PodTemplate: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: image}}},
			},
		}
	}

	annotations := make(map[string]string)
	if provider != "" {
		annotations[consts.KubeAnnotationWorkloadProvider] = provider
	}

	prefill := component("prefill", nvidiacomv1beta1.ComponentTypePrefill, "registry.example/dynamo-worker:1.4.0")
	prefill.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "MODEL_MAX_LEN", Value: "4096"}}
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default", Annotations: annotations},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				prefill,
				component("frontend", nvidiacomv1beta1.ComponentTypeFrontend, "registry.example/dynamo-frontend:1.4.0"),
			},
		},
	}
}
