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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
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
	_, err := validator.Validate(ctx, dgd)
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
					ComponentName: "frontend",
					ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
					Replicas:      k8sptr.To(int32(1)),
				},
				{
					ComponentName: "worker",
					ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
					Replicas:      k8sptr.To(int32(2)),
				},
			},
		},
	}
}

func newAlphaDGDForCompatibilityValidation() *nvidiacomv1alpha1.DynamoGraphDeployment {
	return &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-graph",
			Namespace: "default",
		},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      k8sptr.To(int32(1)),
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
