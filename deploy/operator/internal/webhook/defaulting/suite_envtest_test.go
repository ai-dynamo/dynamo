/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting_test

import (
	"os"
	"reflect"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const defaultingAdmissionOperatorVersion = "1.6.0"

var defaultingAdmissionEnv = operatorenv.New(operatorenv.Options{
	Admission: operatorenv.AdmissionWebhooks{Mutating: true},
	SetupWebhooks: func(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
		return webhooksetup.Setup(mgr, webhooksetup.Options{
			Config:            opts.OperatorConfig,
			RuntimeConfig:     opts.RuntimeConfig,
			OperatorVersion:   opts.OperatorVersion,
			OperatorPrincipal: opts.OperatorPrincipal,
			Gate:              features.Gates{Grove: true},
		})
	},
	OperatorVersion: defaultingAdmissionOperatorVersion,
})

func TestMain(m *testing.M) {
	os.Exit(defaultingAdmissionEnv.RunM(m))
}

func TestDynamoGraphDeploymentDefaultingAdmission(t *testing.T) {
	tests := []struct {
		name                 string
		annotations          map[string]string
		component            nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		wantProvider         string
		wantReplicas         int32
		wantRoleReplicas     []int32
		addComponentOnUpdate bool
	}{
		{
			name: "defaults the complete DGD type tree",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "submitted-value",
				consts.KubeAnnotationWorkloadProvider:            "submitted-value",
			},
			component: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4},
				Roles: []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: nvidiacomv1beta1.ComponentRoleLeader},
					{Name: nvidiacomv1beta1.ComponentRoleWorker},
				},
			},
			wantProvider:         consts.WorkloadProviderGrove,
			wantReplicas:         1,
			wantRoleReplicas:     []int32{1, 3},
			addComponentOnUpdate: true,
		},
		{
			name: "preserves an explicit zero replica count",
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: consts.KubeLabelValueFalse,
			},
			component: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				Replicas:      new(int32(0)),
			},
			wantProvider: consts.WorkloadProviderComponent,
			wantReplicas: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := defaultingAdmissionEnv.ForTest(t)
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				Name: "graph", Namespace: env.Namespace(), Annotations: tt.annotations,
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{tt.component},
				},
			}

			t.Log("Create the DGD through the API server and production mutating handler")
			if err := env.Client().Create(t.Context(), dgd); err != nil {
				t.Fatalf("create DGD: %v", err)
			}
			if got := dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]; got != defaultingAdmissionOperatorVersion {
				t.Fatalf("origin version = %q, want %q", got, defaultingAdmissionOperatorVersion)
			}
			if got := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]; got != tt.wantProvider {
				t.Fatalf("workload provider = %q, want %q", got, tt.wantProvider)
			}
			component := &dgd.Spec.Components[0]
			if got := ptr.Deref(component.Replicas, -1); got != tt.wantReplicas {
				t.Fatalf("component replicas = %d, want %d", got, tt.wantReplicas)
			}
			for i, want := range tt.wantRoleReplicas {
				if got := ptr.Deref(component.Roles[i].Replicas, -1); got != want {
					t.Fatalf("role %d replicas = %d, want %d", i, got, want)
				}
			}
			assertCreateDefaultingIdempotent(t, env.Client(), dgd)

			if tt.addComponentOnUpdate {
				t.Log("Add a component on UPDATE and let structural traversal default the new element")
				dgd.Spec.Components = append(dgd.Spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "decode",
				})
				if err := env.Client().Update(t.Context(), dgd); err != nil {
					t.Fatalf("update DGD: %v", err)
				}
				if got := ptr.Deref(dgd.Spec.Components[1].Replicas, -1); got != 1 {
					t.Fatalf("new component replicas = %d, want 1", got)
				}
			}
		})
	}
}

func TestDynamoComponentDeploymentDefaultingAdmission(t *testing.T) {
	tests := []struct {
		name              string
		componentName     string
		roles             []nvidiacomv1beta1.ComponentRoleSpec
		wantComponentName string
		wantRoleReplicas  []int32
	}{
		{
			name: "defaults the standalone component name and roles",
			roles: []nvidiacomv1beta1.ComponentRoleSpec{
				{Name: nvidiacomv1beta1.ComponentRoleLeader},
				{Name: nvidiacomv1beta1.ComponentRoleWorker},
			},
			wantComponentName: "worker",
			wantRoleReplicas:  []int32{1, 3},
		},
		{
			name:              "preserves explicit component and role values",
			componentName:     "custom",
			roles:             []nvidiacomv1beta1.ComponentRoleSpec{{Name: nvidiacomv1beta1.ComponentRoleLeader, Replicas: new(int32(2))}, {Name: nvidiacomv1beta1.ComponentRoleWorker, Replicas: new(int32(2))}},
			wantComponentName: "custom",
			wantRoleReplicas:  []int32{2, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := defaultingAdmissionEnv.ForTest(t)
			dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
				Name: "worker", Namespace: env.Namespace(),
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					ComponentName: tt.componentName,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4},
					Roles:         tt.roles,
				},
			}

			t.Log("Create the DCD through the API server and production mutating handler")
			if err := env.Client().Create(t.Context(), dcd); err != nil {
				t.Fatalf("create DCD: %v", err)
			}
			if dcd.Spec.ComponentName != tt.wantComponentName {
				t.Fatalf("spec.name = %q, want %q", dcd.Spec.ComponentName, tt.wantComponentName)
			}
			for i, want := range tt.wantRoleReplicas {
				if got := ptr.Deref(dcd.Spec.Roles[i].Replicas, -1); got != want {
					t.Fatalf("role %d replicas = %d, want %d", i, got, want)
				}
			}
			assertCreateDefaultingIdempotent(t, env.Client(), dcd)
		})
	}
}

func TestDynamoGraphDeploymentRequestDefaultingAdmission(t *testing.T) {
	wantDefaultSpec := nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
		Model:          "Qwen/Qwen3-8B",
		Backend:        nvidiacomv1beta1.BackendTypeAuto,
		Image:          "nvcr.io/nvidia/ai-dynamo/dynamo-planner:" + defaultingAdmissionOperatorVersion,
		SearchStrategy: nvidiacomv1beta1.SearchStrategyRapid,
		AutoApply:      new(true),
	}
	tests := []struct {
		name               string
		spec               nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec
		want               nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec
		checkIdempotence   bool
		clearImageOnUpdate bool
	}{
		{
			name: "combines schema and webhook defaults",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "Qwen/Qwen3-8B",
			},
			want:               wantDefaultSpec,
			checkIdempotence:   true,
			clearImageOnUpdate: true,
		},
		{
			name: "treats explicit schema defaults identically",
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model:          "Qwen/Qwen3-8B",
				Backend:        nvidiacomv1beta1.BackendTypeAuto,
				SearchStrategy: nvidiacomv1beta1.SearchStrategyRapid,
				AutoApply:      new(true),
			},
			want: wantDefaultSpec,
		},
		{
			name:             "preserves an explicit profiler image",
			checkIdempotence: true,
			spec: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model: "Qwen/Qwen3-8B",
				Image: "registry.example/planner:custom",
			},
			want: nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec{
				Model:          "Qwen/Qwen3-8B",
				Backend:        nvidiacomv1beta1.BackendTypeAuto,
				Image:          "registry.example/planner:custom",
				SearchStrategy: nvidiacomv1beta1.SearchStrategyRapid,
				AutoApply:      new(true),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := defaultingAdmissionEnv.ForTest(t)
			dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{
				Name: "request", Namespace: env.Namespace(),
				Spec: tt.spec,
			}

			t.Log("Create the DGDR through the API server and production mutating handler")
			if err := env.Client().Create(t.Context(), dgdr); err != nil {
				t.Fatalf("create DGDR: %v", err)
			}
			if !reflect.DeepEqual(dgdr.Spec, tt.want) {
				t.Fatalf("spec = %#v, want %#v", dgdr.Spec, tt.want)
			}
			if tt.checkIdempotence {
				assertCreateDefaultingIdempotent(t, env.Client(), dgdr)
			}

			if tt.clearImageOnUpdate {
				t.Log("Remove the image on UPDATE and verify the CREATE-only default stays absent")
				dgdr.Spec.Image = ""
				if err := env.Client().Update(t.Context(), dgdr); err != nil {
					t.Fatalf("update DGDR: %v", err)
				}
				if dgdr.Spec.Image != "" {
					t.Fatalf("image after UPDATE = %q, want empty", dgdr.Spec.Image)
				}
			}
		})
	}
}

func assertCreateDefaultingIdempotent(t *testing.T, kubeClient client.Client, object client.Object) {
	t.Helper()
	before := defaultingAdmissionState(t, object)
	repeated, ok := object.DeepCopyObject().(client.Object)
	if !ok {
		t.Fatalf("deep copy of %T is not a client.Object", object)
	}
	repeated.SetName(object.GetName() + "-defaulted")
	repeated.SetResourceVersion("")
	repeated.SetUID("")
	repeated.SetGeneration(0)
	repeated.SetCreationTimestamp(metav1.Time{})
	repeated.SetManagedFields(nil)
	if err := kubeClient.Create(t.Context(), repeated); err != nil {
		t.Fatalf("repeat CREATE defaulting: %v", err)
	}
	after := defaultingAdmissionState(t, repeated)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("defaulting is not idempotent:\nbefore: %#v\nafter: %#v", before, after)
	}
}

func defaultingAdmissionState(t *testing.T, object runtime.Object) map[string]any {
	t.Helper()
	value, err := runtime.DefaultUnstructuredConverter.ToUnstructured(object)
	if err != nil {
		t.Fatalf("convert %T to unstructured: %v", object, err)
	}
	metadata, _ := value["metadata"].(map[string]any)
	return map[string]any{
		"annotations": metadata["annotations"],
		"spec":        value["spec"],
	}
}
