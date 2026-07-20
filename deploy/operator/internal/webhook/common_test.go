/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package webhook

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

type testExcludedNamespaces map[string]bool

func (e testExcludedNamespaces) Contains(namespace string) bool {
	return e[namespace]
}

type countingDefaulter struct {
	calls int
}

func (d *countingDefaulter) Default(context.Context, runtime.Object) error {
	d.calls++
	return nil
}

func TestLeaseAwareDefaulter(t *testing.T) {
	defaulter := &countingDefaulter{}
	wrapped := NewLeaseAwareDefaulter(defaulter, testExcludedNamespaces{"claimed": true})

	if err := wrapped.Default(context.Background(), &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: "claimed"},
	}); err != nil {
		t.Fatalf("defaulting excluded namespace: %v", err)
	}
	if defaulter.calls != 0 {
		t.Fatalf("excluded namespace called defaulter %d times, want 0", defaulter.calls)
	}

	if err := wrapped.Default(context.Background(), &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: "unclaimed"},
	}); err != nil {
		t.Fatalf("defaulting unclaimed namespace: %v", err)
	}
	if defaulter.calls != 1 {
		t.Fatalf("unclaimed namespace called defaulter %d times, want 1", defaulter.calls)
	}
}

func TestLeaseAwareDefaulterWithoutChecker(t *testing.T) {
	defaulter := &countingDefaulter{}
	wrapped := NewLeaseAwareDefaulter(defaulter, nil)
	if wrapped != defaulter {
		t.Fatal("defaulter without a checker should be returned unchanged")
	}
}

func TestWithGate(t *testing.T) {
	SetFeatureResolver(features.NewResolver(features.Gates{Grove: true}, func(namespace string) (string, bool) {
		switch namespace {
		case "claimed":
			return `{"grove":false,"futureGate":true}`, true
		case "invalid":
			return `{"grove":"yes"}`, true
		default:
			return "", false
		}
	}))
	t.Cleanup(func() { SetFeatureResolver(nil) })

	webhook := WithGate(&admission.Webhook{Handler: admission.HandlerFunc(func(ctx context.Context, req admission.Request) admission.Response {
		wantGrove := req.Namespace != "claimed"
		if got := features.MustGateFrom(ctx).Enabled(features.Grove); got != wantGrove {
			t.Fatalf("Grove gate = %v, want %v", got, wantGrove)
		}
		return admission.Allowed("handled")
	})}, features.Gates{Grove: true})

	for namespace, wantWarnings := range map[string]int{"claimed": 1, "invalid": 1, "unclaimed": 0} {
		response := webhook.Handler.Handle(context.Background(), admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Namespace: namespace,
		}})
		if !response.Allowed || len(response.Warnings) != wantWarnings {
			t.Fatalf("response for %q = %#v", namespace, response)
		}
	}
}

func TestCanModifyDGDReplicas(t *testing.T) {
	SetNamespaceOperatorPrincipalResolver(func(namespace string) (string, bool) {
		return "system:serviceaccount:tenant-a:dev-operator-controller-manager", namespace == "tenant-a"
	})
	t.Cleanup(func() { SetNamespaceOperatorPrincipalResolver(nil) })

	tests := []struct {
		name          string
		principal     string
		username      string
		expectAllowed bool
	}{
		{
			name:          "operator SA with standard Helm release (dynamo-platform)",
			principal:     "system:serviceaccount:dynamo-system:dynamo-platform-dynamo-operator-controller-manager",
			username:      "system:serviceaccount:dynamo-system:dynamo-platform-dynamo-operator-controller-manager",
			expectAllowed: true,
		},
		{
			name:          "operator SA with collapsed Helm release (dynamo-operator) — the bug scenario",
			principal:     "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			username:      "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			expectAllowed: true,
		},
		{
			name:          "operator SA auto-detected from downward API",
			principal:     "system:serviceaccount:custom-ns:my-release-controller-manager",
			username:      "system:serviceaccount:custom-ns:my-release-controller-manager",
			expectAllowed: true,
		},
		{
			name:          "active namespaced operator SA",
			principal:     "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			username:      "system:serviceaccount:tenant-a:dev-operator-controller-manager",
			expectAllowed: true,
		},
		{
			name:          "operator SA wrong namespace is rejected",
			principal:     "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			username:      "system:serviceaccount:other-ns:dynamo-operator-controller-manager",
			expectAllowed: false,
		},
		{
			name:          "planner SA allowed in any namespace (well-known name)",
			principal:     "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			username:      "system:serviceaccount:user-ns:planner-serviceaccount",
			expectAllowed: true,
		},
		{
			name:          "planner SA allowed with no operator principal set",
			principal:     "",
			username:      "system:serviceaccount:other-ns:planner-serviceaccount",
			expectAllowed: true,
		},
		{
			name:          "unauthorized SA rejected",
			principal:     "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			username:      "system:serviceaccount:user-ns:some-random-sa",
			expectAllowed: false,
		},
		{
			name:          "non-SA user rejected",
			principal:     "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			username:      "admin@example.com",
			expectAllowed: false,
		},
		{
			name:          "malformed SA username rejected",
			principal:     "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			username:      "system:serviceaccount:only-three-parts",
			expectAllowed: false,
		},
		{
			name:          "empty operator principal still permits planner",
			principal:     "",
			username:      "system:serviceaccount:ns:planner-serviceaccount",
			expectAllowed: true,
		},
		{
			name:          "empty operator principal rejects other SA",
			principal:     "",
			username:      "system:serviceaccount:ns:dynamo-operator-controller-manager",
			expectAllowed: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			userInfo := authenticationv1.UserInfo{Username: tt.username}
			got := CanModifyDGDReplicas(tt.principal, userInfo)
			if got != tt.expectAllowed {
				t.Errorf("CanModifyDGDReplicas() = %v, want %v", got, tt.expectAllowed)
			}
		})
	}
}
