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
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	k8sptr "k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
)

func TestDynamoComponentDeploymentV1Alpha1HandlerConvertsRequest(t *testing.T) {
	handler := &dynamoComponentDeploymentV1Alpha1Handler{
		handler: NewDynamoComponentDeploymentHandler(nil, ""),
	}
	ctx := dgdAdmissionContext(admissionv1.Create, nvidiacomv1alpha1.DynamoComponentDeploymentGVK)
	dcd := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ServiceName:            "worker",
				ComponentType:          consts.ComponentTypeWorker,
				RuntimeVersionOverride: "1.1.0",
				ExtraPodSpec:           &nvidiacomv1alpha1.ExtraPodSpec{MainContainer: &corev1.Container{Image: "registry.example/runtime:1.1.0"}},
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
	handler := NewDynamoComponentDeploymentHandler(nil, "")
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

func TestDynamoComponentDeploymentHandlerFencesTransactionalOwnedWorkerReplicas(t *testing.T) {
	t.Log("Build a transactional DGD and its operator-owned worker DCD")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	operatorPrincipal := "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager"
	dgdUID := types.UID("transactional-dgd")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			UID:       dgdUID,
			Annotations: map[string]string{
				nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
			},
		},
	}
	reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd).Build()
	handler := NewDynamoComponentDeploymentHandler(reader, operatorPrincipal)
	controller := true
	oldDCD := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph-decode",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1beta1.GroupVersion.String(),
				Kind:       nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind,
				Name:       dgd.Name,
				UID:        dgdUID,
				Controller: &controller,
			}},
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName:          "decode",
				ComponentType:          nvidiacomv1beta1.ComponentTypeDecode,
				Replicas:               k8sptr.To[int32](1),
				RuntimeVersionOverride: "1.4.0",
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name:  consts.MainContainerName,
					Image: "dynamo:test",
				}}}},
			},
		},
	}
	newDCD := oldDCD.DeepCopy()
	newDCD.Spec.Replicas = k8sptr.To[int32](2)

	t.Log("Reject a non-operator attempt to raise the transactional worker target")
	user := &authenticationv1.UserInfo{Username: "system:serviceaccount:default:replica-writer"}
	ctx := dgdAdmissionContextWithUserInfo(admissionv1.Update, nvidiacomv1beta1.DynamoComponentDeploymentGVK, user)
	if _, err := handler.ValidateUpdate(ctx, oldDCD, newDCD); err == nil ||
		!strings.Contains(err.Error(), "transactional DGD-owned worker replicas are operator-owned") {
		t.Fatalf("ValidateUpdate() error = %v, want transactional replica fence", err)
	}

	t.Log("Reject attaching the transactional owner and raising replicas atomically")
	standaloneOld := oldDCD.DeepCopy()
	standaloneOld.OwnerReferences = nil
	attachedNew := standaloneOld.DeepCopy()
	attachedNew.OwnerReferences = append([]metav1.OwnerReference(nil), oldDCD.OwnerReferences...)
	attachedNew.Spec.Replicas = k8sptr.To[int32](2)
	if _, err := handler.ValidateUpdate(ctx, standaloneOld, attachedNew); err == nil ||
		!strings.Contains(err.Error(), "transactional DGD-owned worker replicas are operator-owned") {
		t.Fatalf("ValidateUpdate() atomic owner attachment error = %v, want transactional replica fence", err)
	}

	t.Log("Allow the exact operator principal to mirror an admitted target")
	operator := &authenticationv1.UserInfo{Username: operatorPrincipal}
	operatorCtx := dgdAdmissionContextWithUserInfo(admissionv1.Update, nvidiacomv1beta1.DynamoComponentDeploymentGVK, operator)
	if _, err := handler.ValidateUpdate(operatorCtx, oldDCD, newDCD); err != nil {
		t.Fatalf("ValidateUpdate() operator error = %v", err)
	}
}
