/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package secret

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestSecretReplicator_Replicate(t *testing.T) {
	sourceSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "source-ns",
		},
		Type: corev1.SecretTypeOpaque,
		Data: map[string][]byte{
			"private.key":     []byte("private-key-content"),
			"private.key.pub": []byte("public-key-content"),
		},
	}

	existingTargetSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "target-ns",
		},
		Type: corev1.SecretTypeOpaque,
		Data: map[string][]byte{
			"private.key":     []byte("existing-private-key"),
			"private.key.pub": []byte("existing-public-key"),
		},
	}

	tests := []struct {
		name              string
		sourceNamespace   string
		secretName        string
		targetNamespace   string
		existingSecrets   []client.Object
		mockGetError      error
		mockCreateError   error
		wantError         bool
		wantErrorContains string
		validateResult    func(t *testing.T, client client.Client)
	}{
		{
			name:            "secret already exists in target namespace - does nothing",
			sourceNamespace: "source-ns",
			secretName:      "test-secret",
			targetNamespace: "target-ns",
			existingSecrets: []client.Object{sourceSecret, existingTargetSecret},
			wantError:       false,
			validateResult: func(t *testing.T, client client.Client) {
				// Should not have modified existing secret
				var secret corev1.Secret
				err := client.Get(context.Background(), types.NamespacedName{
					Name:      "test-secret",
					Namespace: "target-ns",
				}, &secret)
				if err != nil {
					t.Errorf("Expected secret to exist in target namespace")
				}
				if string(secret.Data["private.key"]) != "existing-private-key" {
					t.Errorf("Expected existing secret to remain unchanged")
				}
			},
		},
		{
			name:              "source secret does not exist - returns error",
			sourceNamespace:   "source-ns",
			secretName:        "missing-secret",
			targetNamespace:   "target-ns",
			existingSecrets:   []client.Object{},
			wantError:         true,
			wantErrorContains: "error getting source secret",
		},
		{
			name:            "successful replication",
			sourceNamespace: "source-ns",
			secretName:      "test-secret",
			targetNamespace: "target-ns",
			existingSecrets: []client.Object{sourceSecret},
			wantError:       false,
			validateResult: func(t *testing.T, client client.Client) {
				var secret corev1.Secret
				err := client.Get(context.Background(), types.NamespacedName{
					Name:      "test-secret",
					Namespace: "target-ns",
				}, &secret)
				if err != nil {
					t.Errorf("Expected secret to be created in target namespace: %v", err)
				}
				if secret.Type != corev1.SecretTypeOpaque {
					t.Errorf("Expected secret type %v, got %v", corev1.SecretTypeOpaque, secret.Type)
				}
				if string(secret.Data["private.key"]) != "private-key-content" {
					t.Errorf("Expected private key data to be copied")
				}
				if string(secret.Data["private.key.pub"]) != "public-key-content" {
					t.Errorf("Expected public key data to be copied")
				}
			},
		},
		{
			name:            "race condition - AlreadyExists error is ignored",
			sourceNamespace: "source-ns",
			secretName:      "test-secret",
			targetNamespace: "target-ns",
			existingSecrets: []client.Object{sourceSecret},
			mockCreateError: k8serrors.NewAlreadyExists(schema.GroupResource{Resource: "secrets"}, "test-secret"),
			wantError:       false,
		},
		{
			name:              "create error other than AlreadyExists - returns error",
			sourceNamespace:   "source-ns",
			secretName:        "test-secret",
			targetNamespace:   "target-ns",
			existingSecrets:   []client.Object{sourceSecret},
			mockCreateError:   k8serrors.NewServiceUnavailable("mock error"),
			wantError:         true,
			wantErrorContains: "failed to create replica",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fake client with existing secrets
			scheme := runtime.NewScheme()
			_ = corev1.AddToScheme(scheme)

			clientBuilder := fake.NewClientBuilder().WithScheme(scheme)
			if len(tt.existingSecrets) > 0 {
				clientBuilder = clientBuilder.WithObjects(tt.existingSecrets...)
			}

			fakeClient := clientBuilder.Build()

			// Wrap client to inject errors if needed
			var testClient client.Client = fakeClient
			if tt.mockCreateError != nil {
				testClient = &errorInjectingClient{
					Client:      fakeClient,
					createError: tt.mockCreateError,
				}
			}

			replicator := NewSecretReplicator(testClient, tt.sourceNamespace, tt.secretName)

			err := replicator.Replicate(context.Background(), tt.targetNamespace)

			if tt.wantError {
				if err == nil {
					t.Errorf("Replicate() expected error, got nil")
				} else if tt.wantErrorContains != "" && !strings.Contains(err.Error(), tt.wantErrorContains) {
					t.Errorf("Replicate() error = %v, want error containing %v", err, tt.wantErrorContains)
				}
			} else {
				if err != nil {
					t.Errorf("Replicate() unexpected error = %v", err)
				}
			}

			if tt.validateResult != nil {
				tt.validateResult(t, fakeClient)
			}
		})
	}
}

// errorInjectingClient wraps a client to inject specific errors for testing
type errorInjectingClient struct {
	client.Client
	createError error
}

func (c *errorInjectingClient) Create(ctx context.Context, obj client.Object, opts ...client.CreateOption) error {
	if c.createError != nil {
		return c.createError
	}
	return c.Client.Create(ctx, obj, opts...)
}
