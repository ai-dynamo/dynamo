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
	k8sptr "k8s.io/utils/ptr"
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
		if _, err := handler.ValidateUpdate(ctx, nil, newDGD); err != nil {
			t.Fatalf("ValidateUpdate() error = %v", err)
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

func TestTransactionalAnnotationsImmutable(t *testing.T) {
	handler := NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "system:serviceaccount:dynamo:dynamo-operator")
	createCtx := dgdAdmissionContext(admissionv1.Create, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
	updateCtx := dgdAdmissionContext(admissionv1.Update, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
	annotate := func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
		dgd.Annotations = map[string]string{
			nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
			nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation:   "2400",
			nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation: "2",
		}
	}

	valid := newBetaDGDForValidation()
	annotate(valid)
	if _, err := handler.ValidateCreate(createCtx, valid); err != nil {
		t.Fatalf("ValidateCreate() transactional annotations error = %v", err)
	}

	missingFloor := newBetaDGDForValidation()
	annotate(missingFloor)
	delete(missingFloor.Annotations, nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation)
	if _, err := handler.ValidateCreate(createCtx, missingFloor); err == nil || !strings.Contains(err.Error(), "required when transactional power control is configured") {
		t.Fatalf("ValidateCreate() missing floor error = %v, want required", err)
	}

	for _, key := range []string{
		nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation,
		nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation,
		nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation,
	} {
		t.Run("change "+key, func(t *testing.T) {
			oldDGD := newBetaDGDForValidation()
			annotate(oldDGD)
			newDGD := oldDGD.DeepCopy()
			if key == nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation {
				for annotation := range newDGD.Annotations {
					delete(newDGD.Annotations, annotation)
				}
			} else {
				newDGD.Annotations[key] = "3"
			}
			if _, err := handler.ValidateUpdate(updateCtx, oldDGD, newDGD); err == nil || !strings.Contains(err.Error(), "field is immutable") {
				t.Fatalf("ValidateUpdate() error = %v, want annotation immutability", err)
			}
		})
	}

	oldDGD := newBetaDGDForValidation()
	newDGD := oldDGD.DeepCopy()
	annotate(newDGD)
	if _, err := handler.ValidateUpdate(updateCtx, oldDGD, newDGD); err == nil || !strings.Contains(err.Error(), "field is immutable") {
		t.Fatalf("ValidateUpdate() add mode error = %v, want annotation immutability", err)
	}
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

func TestNoDGPBWebhook(t *testing.T) {
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

	for _, path := range []string{
		"/validate/nvidia.com/v1beta1/dynamographpowerbudgets",
		"/mutate/nvidia.com/v1beta1/dynamographpowerbudgets",
		"/validate-nvidia-com-v1beta1-dynamographpowerbudget",
		"/mutate-nvidia-com-v1beta1-dynamographpowerbudget",
	} {
		request := httptest.NewRequest("POST", path, nil)
		_, pattern := server.WebhookMux().Handler(request)
		if pattern != "" {
			t.Fatalf("DGPB webhook registered for %q as %q", path, pattern)
		}
	}
}

func TestNoV1Alpha1Webhook(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	server := ctrlwebhook.NewServer(ctrlwebhook.Options{})
	mgr := &fakeManager{scheme: scheme, webhookServer: server}
	if err := NewDynamoGraphDeploymentHandler(mgr, "").RegisterWithManager(mgr, features.Defaults()); err != nil {
		t.Fatalf("RegisterWithManager() error = %v", err)
	}
	request := httptest.NewRequest("POST", "/validate-nvidia-com-v1alpha1-dynamographdeployment", nil)
	if _, pattern := server.WebhookMux().Handler(request); pattern != "" {
		t.Fatalf("v1alpha1 DGD webhook registered as %q", pattern)
	}
}

func TestCheckpointIncompatible(t *testing.T) {
	handler := NewDynamoGraphDeploymentHandler(newGroveTopologyTestManager(t), "")
	ctx := dgdAdmissionContext(admissionv1.Create, nvidiacomv1beta1.DynamoGraphDeploymentGVK)
	transactionalDGD := func() *nvidiacomv1beta1.DynamoGraphDeployment {
		dgd := newBetaDGDForValidation()
		dgd.Annotations = map[string]string{
			nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
			nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation:   "2400",
			nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation: "1",
		}
		return dgd
	}

	for _, tc := range []struct {
		name       string
		checkpoint *nvidiacomv1beta1.ComponentCheckpointConfig
		wantError  bool
	}{
		{name: "enabled", checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}, wantError: true},
		{name: "checkpointRef", checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{CheckpointRef: k8sptr.To("saved")}, wantError: true},
		{name: "disabled", checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: false}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dgd := transactionalDGD()
			dgd.GetComponentByName("worker").Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: tc.checkpoint}
			_, err := handler.ValidateCreate(ctx, dgd)
			if tc.wantError {
				if err == nil || !strings.Contains(err.Error(), "spec.components[1].experimental.checkpoint: Forbidden") {
					t.Fatalf("ValidateCreate() error = %v, want hub checkpoint incompatibility", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateCreate() disabled checkpoint error = %v", err)
			}
		})
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
