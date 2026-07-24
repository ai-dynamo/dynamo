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
	"net/http/httptest"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
)

func TestDynamoComponentDeploymentV1Alpha1HandlerConvertsRequest(t *testing.T) {
	handler := &dynamoComponentDeploymentV1Alpha1Handler{
		handler: NewDynamoComponentDeploymentHandler(),
	}
	ctx := dgdAdmissionContext(admissionv1.Create, nvidiacomv1alpha1.DynamoComponentDeploymentGVK)
	dcd := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ServiceName:   "worker",
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}

	warnings, err := handler.ValidateCreate(ctx, dcd)
	if err != nil {
		t.Fatalf("ValidateCreate() error = %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("ValidateCreate() warnings = %v, want none", warnings)
	}
}

func TestCastToDynamoComponentDeployment(t *testing.T) {
	beta := &nvidiacomv1beta1.DynamoComponentDeployment{
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
			},
		},
	}
	got, err := castToDynamoComponentDeployment(beta)
	if err != nil || got != beta {
		t.Fatalf("castToDynamoComponentDeployment() = (%v, %v), want original DCD", got, err)
	}

	alpha := &nvidiacomv1alpha1.DynamoComponentDeployment{
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ServiceName: "worker",
			},
		},
	}
	got, err = castToDynamoComponentDeployment(alpha)
	if err != nil {
		t.Fatalf("castToDynamoComponentDeployment() error = %v", err)
	}
	if got.Spec.ComponentName != alpha.Spec.ServiceName {
		t.Fatalf("converted component name = %q, want %q", got.Spec.ComponentName, alpha.Spec.ServiceName)
	}

	if _, err := castToDynamoComponentDeployment(nil); err == nil {
		t.Fatal("castToDynamoComponentDeployment() error = nil, want type mismatch")
	}
}

func TestDynamoComponentDeploymentHandlerRegisterWithManager(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 scheme: %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}

	server := ctrlwebhook.NewServer(ctrlwebhook.Options{})
	mgr := &fakeManager{scheme: scheme, webhookServer: server}
	handler := NewDynamoComponentDeploymentHandler()
	if err := handler.RegisterWithManager(mgr, features.Defaults()); err != nil {
		t.Fatalf("RegisterWithManager() error = %v", err)
	}

	for _, path := range []string{
		dynamoComponentDeploymentV1Alpha1WebhookPath,
		dynamoComponentDeploymentV1Beta1WebhookPath,
	} {
		request := httptest.NewRequest("POST", path, nil)
		_, pattern := server.WebhookMux().Handler(request)
		if pattern != path {
			t.Fatalf("registered pattern = %q, want %q", pattern, path)
		}
	}
}
