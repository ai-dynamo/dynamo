/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/client-go/rest"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
)

const sglangBackendFramework = "sglang"

func TestDynamoGraphDeploymentConversionFailureIsFatal(t *testing.T) {
	dgd := newBetaDGDForValidation()
	dgd.Spec.Components = append(dgd.Spec.Components, dgd.Spec.Components[0])

	validator := newDynamoGraphDeploymentTestValidator(t)
	ctx := features.WithGate(context.Background(), features.Gates{Grove: true})
	_, err := validator.Validate(ctx, dgd, runtimeVersionSourceV1Beta1)
	if err == nil || !strings.Contains(err.Error(), "failed to reconstruct compatibility view") {
		t.Fatalf("Validate() error = %v, want fatal conversion error", err)
	}
	if k8serrors.IsInvalid(err) {
		t.Fatalf("Validate() error = %v, want fatal conversion error rather than field validation error", err)
	}
}

func assertFieldPaths(t *testing.T, errs field.ErrorList, want []string) {
	t.Helper()
	got := make([]string, len(errs))
	for i := range errs {
		got[i] = errs[i].Field
	}
	if !slices.Equal(got, want) {
		t.Fatalf("field paths = %v, want %v", got, want)
	}
}

func newBetaDGDForValidation() *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-graph",
			Namespace: "default",
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:          "frontend",
					ComponentType:          nvidiacomv1beta1.ComponentTypeFrontend,
					RuntimeVersionOverride: "1.1.0",
					Replicas:               k8sptr.To(int32(1)),
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
					}},
				},
				{
					ComponentName:          "worker",
					ComponentType:          nvidiacomv1beta1.ComponentTypeWorker,
					RuntimeVersionOverride: "1.1.0",
					Replicas:               k8sptr.To(int32(2)),
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
					}},
				},
			},
		},
	}
}

type fakeManager struct {
	ctrl.Manager
	client        client.Client
	config        *rest.Config
	scheme        *runtime.Scheme
	webhookServer ctrlwebhook.Server
}

func (m *fakeManager) GetClient() client.Client             { return m.client }
func (m *fakeManager) GetConfig() *rest.Config              { return m.config }
func (m *fakeManager) GetScheme() *runtime.Scheme           { return m.scheme }
func (m *fakeManager) GetWebhookServer() ctrlwebhook.Server { return m.webhookServer }

func newDynamoGraphDeploymentTestValidator(t *testing.T) *DynamoGraphDeploymentValidator {
	t.Helper()
	return NewDynamoGraphDeploymentValidator(newGroveTopologyTestManager(t))
}

func newGroveTopologyTestManager(t *testing.T) ctrl.Manager {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := grovev1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add Grove scheme: %v", err)
	}
	return &fakeManager{
		client: fake.NewClientBuilder().WithScheme(scheme).Build(),
		config: &rest.Config{},
	}
}

func assertBetaValidationErrors(t *testing.T, err error, wantErrs []string) {
	t.Helper()
	if len(wantErrs) == 0 {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("expected errors %v but got nil", wantErrs)
	}
	statusErr, ok := err.(*k8serrors.StatusError)
	if !ok || !k8serrors.IsInvalid(err) {
		t.Fatalf("error = %T %v, want typed Kubernetes invalid error", err, err)
	}
	if statusErr.ErrStatus.Details == nil {
		t.Fatalf("error = %v, want typed field causes", err)
	}

	causes := statusErr.ErrStatus.Details.Causes
	gotErrs := make([]string, len(causes))
	for i, cause := range causes {
		if cause.Field == "" {
			t.Fatalf("error cause = %#v, want an exact field path", cause)
		}
		gotErrs[i] = fmt.Sprintf("%s: %s", cause.Field, cause.Message)
	}
	if !slices.Equal(gotErrs, wantErrs) {
		t.Fatalf("webhook errors = %v, want %v", gotErrs, wantErrs)
	}
}

func elasticEPSharedSpec(command, args []string) *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	return &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    consts.MainContainerName,
				Command: command,
				Args:    args,
			}},
		}},
	}
}

func TestValidateElasticEPRequiresCommand(t *testing.T) {
	const vllm = "vllm"
	rayArgs := []string{"--model", "test", "--data-parallel-backend", "ray", "--enable-elastic-ep"}
	fldPath := field.NewPath("spec")
	const commandPath = "spec.podTemplate.spec.containers[0].command"

	tests := []struct {
		name    string
		backend string
		spec    *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		want    []string
	}{
		{
			name:    "vllm elastic-EP ray with empty command is rejected",
			backend: vllm,
			spec:    elasticEPSharedSpec(nil, rayArgs),
			want:    []string{commandPath},
		},
		{
			name:    "vllm elastic-EP with -dpb=ray alias and empty command is rejected",
			backend: vllm,
			spec:    elasticEPSharedSpec(nil, []string{"--model", "test", "-dpb=ray", "--enable-elastic-ep"}),
			want:    []string{commandPath},
		},
		{
			name:    "explicit command is accepted",
			backend: vllm,
			spec:    elasticEPSharedSpec([]string{"python3", "-m", "dynamo.vllm"}, rayArgs),
			want:    nil,
		},
		{
			name:    "elastic-EP flags carried in Command are accepted",
			backend: vllm,
			spec:    elasticEPSharedSpec([]string{"python3", "-m", "dynamo.vllm", "--data-parallel-backend", "ray", "--enable-elastic-ep"}, nil),
			want:    nil,
		},
		{
			name:    "non-vllm backend is not validated",
			backend: sglangBackendFramework,
			spec:    elasticEPSharedSpec(nil, rayArgs),
			want:    nil,
		},
		{
			name:    "vllm without elastic-EP is accepted",
			backend: vllm,
			spec:    elasticEPSharedSpec(nil, []string{"--model", "test"}),
			want:    nil,
		},
		{
			name:    "vllm elastic-EP on a non-ray backend is accepted",
			backend: vllm,
			spec:    elasticEPSharedSpec(nil, []string{"--model", "test", "--data-parallel-backend", "mp", "--enable-elastic-ep"}),
			want:    nil,
		},
		{
			name:    "nil pod template is ignored",
			backend: vllm,
			spec:    &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{},
			want:    nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertFieldPaths(t, validateElasticEPRequiresCommand(tt.backend, tt.spec, fldPath), tt.want)
		})
	}
}

// TestDynamoGraphDeploymentRejectsElasticEPWithoutCommand proves the rule is
// wired into the DGD admission path end to end, not just callable in isolation.
func TestDynamoGraphDeploymentRejectsElasticEPWithoutCommand(t *testing.T) {
	dgd := newBetaDGDForValidation()
	// components[1] is the worker; make it request elastic-EP Ray with no command.
	dgd.Spec.Components[1].PodTemplate.Spec.Containers[0].Args = []string{
		"--model", "test", "--data-parallel-backend", "ray", "--enable-elastic-ep",
	}

	validator := newDynamoGraphDeploymentTestValidator(t)
	ctx := features.WithGate(context.Background(), features.Gates{Grove: true})
	_, err := validator.Validate(ctx, dgd, runtimeVersionSourceV1Beta1)
	if err == nil || !k8serrors.IsInvalid(err) {
		t.Fatalf("Validate() error = %v, want invalid field error", err)
	}
	if !strings.Contains(err.Error(), "requires an explicit container command") {
		t.Fatalf("Validate() error = %v, want elastic-EP command requirement", err)
	}
}
