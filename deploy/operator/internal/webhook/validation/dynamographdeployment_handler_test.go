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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func TestDynamoGraphDeploymentHandlerValidateCreate(t *testing.T) {
	handler := NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "system:serviceaccount:dynamo:dynamo-operator")
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

}

func TestDynamoGraphDeploymentHandlerValidateUpdate(t *testing.T) {
	handler := NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "system:serviceaccount:dynamo:dynamo-operator")
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
		// Admission always supplies the old object on UPDATE, and the
		// terminating path now reads it to enforce durable metadata rules.
		if _, err := handler.ValidateUpdate(ctx, oldDGD, newDGD); err != nil {
			t.Fatalf("ValidateUpdate() error = %v", err)
		}
	})

	// A finalizer can hold a DGD terminating for an arbitrary period. Durable
	// controller-owned metadata still has to be protected during that window,
	// while anything required for cleanup to finish must still be allowed.
	t.Run("terminating rejects replacing the workload provider", func(t *testing.T) {
		t.Log("given a terminating DGD whose workload provider is already materialized")
		oldDGD := newBetaDGDForValidation()
		oldDGD.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove}
		newDGD := oldDGD.DeepCopy()
		now := metav1.Now()
		newDGD.DeletionTimestamp = &now
		newDGD.Annotations[consts.KubeAnnotationWorkloadProvider] = consts.WorkloadProviderComponent

		t.Log("when the update replaces that provider")
		_, err := handler.ValidateUpdate(ctx, oldDGD, newDGD)

		t.Log("then the annotation is reported immutable")
		assertBetaValidationErrors(t, err, []string{
			`metadata.annotations[nvidia.com/workload-provider]: Invalid value: "component": field is immutable`,
		})
	})

	t.Run("terminating rejects removing the workload provider", func(t *testing.T) {
		t.Log("given a terminating DGD whose workload provider is already materialized")
		oldDGD := newBetaDGDForValidation()
		oldDGD.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove}
		newDGD := oldDGD.DeepCopy()
		now := metav1.Now()
		newDGD.DeletionTimestamp = &now
		delete(newDGD.Annotations, consts.KubeAnnotationWorkloadProvider)

		t.Log("when the update removes that provider")
		_, err := handler.ValidateUpdate(ctx, oldDGD, newDGD)

		t.Log("then the annotation is reported immutable")
		assertBetaValidationErrors(t, err, []string{
			// Removal reports a null bad value, since there is no new value to name.
			"metadata.annotations[nvidia.com/workload-provider]: Invalid value: null: field is immutable",
		})
	})

	t.Run("terminating rejects an unsupported workload provider", func(t *testing.T) {
		t.Log("given the operator identity materializing a provider on a terminating DGD")
		operatorCtx := dgdAdmissionContextWithUserInfo(
			admissionv1.Update,
			nvidiacomv1beta1.DynamoGraphDeploymentGVK,
			&authenticationv1.UserInfo{Username: "system:serviceaccount:dynamo:dynamo-operator"},
		)
		oldDGD := newBetaDGDForValidation()
		newDGD := oldDGD.DeepCopy()
		now := metav1.Now()
		newDGD.DeletionTimestamp = &now
		newDGD.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: "bogus"}

		t.Log("when the value names no program the controller implements")
		_, err := handler.ValidateUpdate(operatorCtx, oldDGD, newDGD)

		t.Log("then the value is reported unsupported")
		assertBetaValidationErrors(t, err, []string{
			`metadata.annotations[nvidia.com/workload-provider]: Unsupported value: "bogus": supported values: "component", "grove"`,
		})
	})

	t.Run("terminating accepts a finalizer-only update", func(t *testing.T) {
		t.Log("given a terminating DGD holding a finalizer")
		oldDGD := newBetaDGDForValidation()
		oldDGD.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove}
		oldDGD.Finalizers = []string{"nvidia.com/dynamo-graph-deployment-finalizer"}
		newDGD := oldDGD.DeepCopy()
		now := metav1.Now()
		newDGD.DeletionTimestamp = &now
		newDGD.Finalizers = nil

		t.Log("when the update only drops that finalizer")
		t.Log("then cleanup is allowed to proceed")
		if _, err := handler.ValidateUpdate(ctx, oldDGD, newDGD); err != nil {
			t.Fatalf("ValidateUpdate() on finalizer removal error = %v, want nil", err)
		}
	})

	// A legacy object whose existing settings no longer satisfy today's rules has
	// to stay deletable. The GPU memory service traversal on the update path
	// judges the new component on its own, so running it here would reject the
	// controller's finalizer-removal update and strand the object forever.
	t.Run("terminating accepts finalizer removal despite an invalid gpu memory service", func(t *testing.T) {
		t.Log("given a terminating DGD whose existing GPU memory service block fails today's rules")
		oldDGD := newBetaDGDForValidation()
		worker := &oldDGD.Spec.Components[1]
		worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
			GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
				ExtraClientContainers: []string{"no-such-container"},
			},
		}
		oldDGD.Finalizers = []string{"nvidia.com/dynamo-graph-deployment-finalizer"}
		newDGD := oldDGD.DeepCopy()
		now := metav1.Now()
		newDGD.DeletionTimestamp = &now
		newDGD.Finalizers = nil

		t.Log("when the update only drops the finalizer")
		t.Log("then no new-state rule is allowed to strand the object")
		if _, err := handler.ValidateUpdate(ctx, oldDGD, newDGD); err != nil {
			t.Fatalf("ValidateUpdate() on finalizer removal error = %v, want nil", err)
		}
	})

	t.Run("stateless validation failure", func(t *testing.T) {
		invalid := newBetaDGDForValidation()
		invalid.Spec.Components = nil
		_, err := handler.ValidateUpdate(ctx, newBetaDGDForValidation(), invalid)
		assertBetaValidationErrors(t, err, []string{"spec.components: Required value: must have at least one component"})
	})

	t.Run("stateful validation failure", func(t *testing.T) {
		oldDGD := newBetaDGDForValidation()
		newDGD := oldDGD.DeepCopy()
		oldDGD.Spec.BackendFramework = "vllm"
		newDGD.Spec.BackendFramework = sglangBackendFramework
		_, err := handler.ValidateUpdate(ctx, oldDGD, newDGD)
		assertBetaValidationErrors(t, err, []string{`spec.backendFramework: Invalid value: "sglang": is immutable and cannot be changed after creation`})
	})

	t.Run("missing operator identity does not block legacy provider materialization", func(t *testing.T) {
		oldDGD := newBetaDGDForValidation()
		newDGD := oldDGD.DeepCopy()
		newDGD.Annotations = map[string]string{
			consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent,
		}
		unconfiguredHandler := NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "")
		if _, err := unconfiguredHandler.ValidateUpdate(ctx, oldDGD, newDGD); err != nil {
			t.Fatalf("ValidateUpdate() error = %v, want optional operator identity to remain permissive", err)
		}
	})
}

func TestDynamoGraphDeploymentHandlerValidateDelete(t *testing.T) {
	handler := NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "")
	ctx := dgdAdmissionContext(admissionv1.Delete, nvidiacomv1beta1.DynamoGraphDeploymentGVK)

	warnings, err := handler.ValidateDelete(ctx, newBetaDGDForValidation())
	if err != nil {
		t.Fatalf("ValidateDelete() error = %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("ValidateDelete() warnings = %v, want none", warnings)
	}

}

func TestDynamoGraphDeploymentHandlerRegisterWithManager(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}

	server := ctrlwebhook.NewServer(ctrlwebhook.Options{})
	mgr := &fakeManager{scheme: scheme, webhookServer: server}
	handler := NewDynamoGraphDeploymentHandler(mgr, "")
	if err := handler.RegisterWithManager(mgr, features.Defaults()); err != nil {
		t.Fatalf("RegisterWithManager() error = %v", err)
	}

	for _, tc := range []struct {
		path        string
		wantPattern string
	}{
		{path: dynamoGraphDeploymentWebhookPath, wantPattern: dynamoGraphDeploymentWebhookPath},
		{path: "/validate-nvidia-com-v1alpha1-dynamographdeployment"},
	} {
		request := httptest.NewRequest("POST", tc.path, nil)
		_, pattern := server.WebhookMux().Handler(request)
		if pattern != tc.wantPattern {
			t.Fatalf("registered pattern for %q = %q, want %q", tc.path, pattern, tc.wantPattern)
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
	ctx := admission.NewContextWithRequest(context.Background(), admission.Request{
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
	return features.WithGate(ctx, features.Defaults())
}
