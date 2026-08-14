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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"k8s.io/apimachinery/pkg/runtime"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
)

func TestDynamoComponentDeploymentHandlerRegisterWithManager(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}

	server := ctrlwebhook.NewServer(ctrlwebhook.Options{})
	mgr := &fakeManager{scheme: scheme, webhookServer: server}
	handler := NewDynamoComponentDeploymentHandler()
	if err := handler.RegisterWithManager(mgr, features.Defaults()); err != nil {
		t.Fatalf("RegisterWithManager() error = %v", err)
	}

	for _, tc := range []struct {
		path        string
		wantPattern string
	}{
		{
			path:        dynamoComponentDeploymentV1Beta1WebhookPath,
			wantPattern: dynamoComponentDeploymentV1Beta1WebhookPath,
		},
		{path: "/validate-nvidia-com-v1alpha1-dynamocomponentdeployment"},
	} {
		request := httptest.NewRequest("POST", tc.path, nil)
		_, pattern := server.WebhookMux().Handler(request)
		if pattern != tc.wantPattern {
			t.Fatalf("registered pattern for %q = %q, want %q", tc.path, pattern, tc.wantPattern)
		}
	}
}
