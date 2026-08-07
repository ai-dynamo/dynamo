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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
)

func TestDynamoComponentDeploymentV1Alpha1HandlerConvertsRequest(t *testing.T) {
	handler := &dynamoComponentDeploymentV1Alpha1Handler{
		handler: NewDynamoComponentDeploymentHandler(""),
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

func TestDynamoComponentDeploymentHandlerCheckpointFailoverBoundary(t *testing.T) {
	const principal = "system:serviceaccount:dynamo-system:dynamo-operator"
	controller := true
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1beta1.GroupVersion.String(),
				Kind:       "DynamoGraphDeployment",
				Name:       "graph",
				UID:        types.UID("graph-uid"),
				Controller: &controller,
			}},
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName:          "worker",
				ComponentType:          nvidiacomv1beta1.ComponentTypeWorker,
				RuntimeVersionOverride: "1.1.0",
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  consts.MainContainerName,
						Image: "registry.example/runtime:1.1.0",
						Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
							corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
						}},
					}},
				}},
				Experimental: &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled:       true,
						CheckpointRef: ptr.To("checkpoint"),
					},
					GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
						Mode: nvidiacomv1beta1.GMSModeIntraPod,
					},
					Failover: &nvidiacomv1beta1.FailoverSpec{
						Mode:       nvidiacomv1beta1.GMSModeIntraPod,
						NumShadows: 1,
					},
				},
			},
		},
	}
	gates := features.Defaults()
	gates.Checkpoint = true
	validate := func(username string) error {
		ctx := dgdAdmissionContextWithUserInfo(
			admissionv1.Create,
			nvidiacomv1beta1.DynamoComponentDeploymentGVK,
			&authenticationv1.UserInfo{Username: username},
		)
		ctx = features.WithGate(ctx, gates)
		_, err := NewDynamoComponentDeploymentHandler(principal).ValidateCreate(ctx, dcd)
		return err
	}

	if err := validate(principal); err != nil {
		t.Fatalf("operator-generated DCD was rejected: %v", err)
	}
	if err := validate("user"); err == nil || !strings.Contains(err.Error(), "only supported for an operator-generated DCD") {
		t.Fatalf("standalone DCD error = %v, want operator-generated rejection", err)
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
	handler := NewDynamoComponentDeploymentHandler("")
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
