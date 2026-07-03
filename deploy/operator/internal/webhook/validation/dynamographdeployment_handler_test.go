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

package validation

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func TestDynamoGraphDeploymentV1Alpha1Handler(t *testing.T) {
	handler := &dynamoGraphDeploymentV1Alpha1Handler{
		handler: NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "", false),
	}

	t.Run("create", func(t *testing.T) {
		dgd := newAlphaDGDForCompatibilityValidation()
		dgd.Spec.Services["worker"].Annotations = map[string]string{
			consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
		}

		_, err := handler.ValidateCreate(
			dgdAdmissionContext(admissionv1.Create, nvidiacomv1alpha1.DynamoGraphDeploymentGVK),
			dgd,
		)
		if err == nil {
			t.Fatal("ValidateCreate() error = nil, want preserved v1alpha1 validation error")
		}
		if !strings.Contains(err.Error(), "spec.services[worker].annotations[nvidia.com/vllm-distributed-executor-backend]") {
			t.Fatalf("ValidateCreate() error = %q, want v1alpha1 service annotation validation error", err)
		}
	})

	t.Run("update", func(t *testing.T) {
		oldDGD := newAlphaDGDForCompatibilityValidation()
		newDGD := oldDGD.DeepCopy()
		newDGD.Labels = map[string]string{"updated": "true"}
		warnings, err := handler.ValidateUpdate(
			dgdAdmissionContext(admissionv1.Update, nvidiacomv1alpha1.DynamoGraphDeploymentGVK),
			oldDGD,
			newDGD,
		)
		if err != nil || len(warnings) != 0 {
			t.Fatalf("ValidateUpdate() = (%v, %v), want no warnings or error", warnings, err)
		}
	})

	t.Run("delete", func(t *testing.T) {
		warnings, err := handler.ValidateDelete(
			dgdAdmissionContext(admissionv1.Delete, nvidiacomv1alpha1.DynamoGraphDeploymentGVK),
			newAlphaDGDForCompatibilityValidation(),
		)
		if err != nil || len(warnings) != 0 {
			t.Fatalf("ValidateDelete() = (%v, %v), want no warnings or error", warnings, err)
		}
	})
}

func TestDynamoGraphDeploymentHandlerValidateCreate(t *testing.T) {
	handler := NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "system:serviceaccount:dynamo:dynamo-operator", false)
	dgd := newBetaDGDForValidation()

	warnings, err := handler.ValidateCreate(dgdAdmissionContext(admissionv1.Create, nvidiacomv1beta1.DynamoGraphDeploymentGVK), dgd)
	if err != nil {
		t.Fatalf("ValidateCreate() error = %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("ValidateCreate() warnings = %v, want none", warnings)
	}

	_, err = handler.ValidateCreate(context.Background(), dgd)
	if err == nil || !strings.Contains(err.Error(), "admission request missing from context") {
		t.Fatalf("ValidateCreate() error = %v, want missing admission request", err)
	}

	_, err = handler.ValidateCreate(
		dgdAdmissionContext(admissionv1.Create, schema.GroupVersionKind{Group: "wrong.example.com", Version: "v1", Kind: "Wrong"}),
		dgd,
	)
	if err == nil || !strings.Contains(err.Error(), "admission requires") {
		t.Fatalf("ValidateCreate() error = %v, want GVK mismatch", err)
	}

	_, err = handler.ValidateCreate(
		dgdAdmissionContext(admissionv1.Create, nvidiacomv1beta1.DynamoGraphDeploymentGVK),
		&runtime.Unknown{},
	)
	if err == nil || !strings.Contains(err.Error(), "expected v1alpha1 or v1beta1 DynamoGraphDeployment") {
		t.Fatalf("ValidateCreate() error = %v, want type mismatch", err)
	}
}

func TestDynamoGraphDeploymentHandlerValidateUpdate(t *testing.T) {
	handler := NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "system:serviceaccount:dynamo:dynamo-operator", false)
	ctx := dgdAdmissionContext(admissionv1.Update, nvidiacomv1beta1.DynamoGraphDeploymentGVK)

	t.Run("valid", func(t *testing.T) {
		oldDGD := newBetaDGDForValidation()
		newDGD := oldDGD.DeepCopy()
		warnings, err := handler.ValidateUpdate(ctx, oldDGD, newDGD)
		if err != nil {
			t.Fatalf("ValidateUpdate() error = %v", err)
		}
		if len(warnings) != 0 {
			t.Fatalf("ValidateUpdate() warnings = %v, want none", warnings)
		}
	})

	t.Run("deleting", func(t *testing.T) {
		oldDGD := newBetaDGDForValidation()
		newDGD := oldDGD.DeepCopy()
		now := metav1.Now()
		newDGD.DeletionTimestamp = &now
		if _, err := handler.ValidateUpdate(ctx, &runtime.Unknown{}, newDGD); err != nil {
			t.Fatalf("ValidateUpdate() error = %v", err)
		}
	})

	t.Run("invalid new object", func(t *testing.T) {
		_, err := handler.ValidateUpdate(ctx, newBetaDGDForValidation(), &runtime.Unknown{})
		if err == nil || !strings.Contains(err.Error(), "expected v1alpha1 or v1beta1 DynamoGraphDeployment") {
			t.Fatalf("ValidateUpdate() error = %v, want new object type mismatch", err)
		}
	})

	t.Run("invalid old object", func(t *testing.T) {
		_, err := handler.ValidateUpdate(ctx, &runtime.Unknown{}, newBetaDGDForValidation())
		if err == nil || !strings.Contains(err.Error(), "expected v1alpha1 or v1beta1 DynamoGraphDeployment") {
			t.Fatalf("ValidateUpdate() error = %v, want old object type mismatch", err)
		}
	})

	t.Run("stateless validation failure", func(t *testing.T) {
		invalid := newBetaDGDForValidation()
		invalid.Spec.Components = nil
		_, err := handler.ValidateUpdate(ctx, newBetaDGDForValidation(), invalid)
		assertBetaValidationErrors(t, err, []string{"spec.components must have at least one component"})
	})

	t.Run("stateful validation failure", func(t *testing.T) {
		oldDGD := newBetaDGDForValidation()
		newDGD := oldDGD.DeepCopy()
		oldDGD.Spec.BackendFramework = "vllm"
		newDGD.Spec.BackendFramework = sglangBackendFramework
		_, err := handler.ValidateUpdate(ctx, oldDGD, newDGD)
		assertBetaValidationErrors(t, err, []string{"spec.backendFramework is immutable and cannot be changed after creation"})
	})
}

func TestDynamoGraphDeploymentHandlerValidateDelete(t *testing.T) {
	handler := NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "", false)
	ctx := dgdAdmissionContext(admissionv1.Delete, nvidiacomv1beta1.DynamoGraphDeploymentGVK)

	warnings, err := handler.ValidateDelete(ctx, newBetaDGDForValidation())
	if err != nil {
		t.Fatalf("ValidateDelete() error = %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("ValidateDelete() warnings = %v, want none", warnings)
	}

	_, err = handler.ValidateDelete(ctx, &runtime.Unknown{})
	if err == nil || !strings.Contains(err.Error(), "expected DynamoGraphDeployment") {
		t.Fatalf("ValidateDelete() error = %v, want type mismatch", err)
	}
}

func TestCastToDynamoGraphDeployment(t *testing.T) {
	dgd := newBetaDGDForValidation()
	got, err := castToDynamoGraphDeployment(dgd)
	if err != nil || got != dgd {
		t.Fatalf("castToDynamoGraphDeployment() = (%v, %v), want original DGD", got, err)
	}

	if _, err := castToDynamoGraphDeployment(nil); err == nil {
		t.Fatal("castToDynamoGraphDeployment() error = nil, want type mismatch")
	}
}

func TestDynamoGraphDeploymentHandlerRegisterWithManager(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 scheme: %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}

	server := ctrlwebhook.NewServer(ctrlwebhook.Options{})
	mgr := &fakeManager{scheme: scheme, webhookServer: server}
	handler := NewDynamoGraphDeploymentHandler(mgr, "", false)
	if err := handler.RegisterWithManager(mgr); err != nil {
		t.Fatalf("RegisterWithManager() error = %v", err)
	}

	for _, path := range []string{
		dynamoGraphDeploymentV1Alpha1WebhookPath,
		dynamoGraphDeploymentV1Beta1WebhookPath,
	} {
		request := httptest.NewRequest("POST", path, nil)
		_, pattern := server.WebhookMux().Handler(request)
		if pattern != path {
			t.Fatalf("registered pattern = %q, want %q", pattern, path)
		}
	}
}

func dgdAdmissionContext(operation admissionv1.Operation, gvk schema.GroupVersionKind) context.Context {
	return dgdAdmissionContextWithUserInfo(operation, gvk, nil)
}

func dgdAdmissionContextWithUserInfo(
	operation admissionv1.Operation,
	gvk schema.GroupVersionKind,
	userInfo *authenticationv1.UserInfo,
) context.Context {
	requestUserInfo := authenticationv1.UserInfo{}
	if userInfo != nil {
		requestUserInfo = *userInfo.DeepCopy()
	}
	return admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: operation,
			UserInfo:  requestUserInfo,
			Kind: metav1.GroupVersionKind{
				Group:   gvk.Group,
				Version: gvk.Version,
				Kind:    gvk.Kind,
			},
		},
	})
}
