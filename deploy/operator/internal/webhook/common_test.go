/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package webhook

import (
	"testing"

	authenticationv1 "k8s.io/api/authentication/v1"
)

func TestCanModifyDGDReplicas(t *testing.T) {
	tests := []struct {
		name             string
		allowedModifiers []string
		username         string
		expectAllowed    bool
	}{
		{
			name:             "operator SA with standard Helm release (dynamo-platform)",
			allowedModifiers: []string{"system:serviceaccount:dynamo-system:dynamo-platform-dynamo-operator-controller-manager"},
			username:         "system:serviceaccount:dynamo-system:dynamo-platform-dynamo-operator-controller-manager",
			expectAllowed:    true,
		},
		{
			name:             "operator SA with collapsed Helm release (dynamo-operator) — the bug scenario",
			allowedModifiers: []string{"system:serviceaccount:dynamo-system:dynamo-operator-controller-manager"},
			username:         "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			expectAllowed:    true,
		},
		{
			name:             "operator SA auto-detected from JWT",
			allowedModifiers: []string{"system:serviceaccount:custom-ns:my-release-controller-manager"},
			username:         "system:serviceaccount:custom-ns:my-release-controller-manager",
			expectAllowed:    true,
		},
		{
			name:             "operator SA wrong namespace is rejected (full principal check)",
			allowedModifiers: []string{"system:serviceaccount:dynamo-system:dynamo-operator-controller-manager"},
			username:         "system:serviceaccount:other-ns:dynamo-operator-controller-manager",
			expectAllowed:    false,
		},
		{
			name:             "planner SA allowed in any namespace (well-known name)",
			allowedModifiers: []string{"system:serviceaccount:dynamo-system:dynamo-operator-controller-manager"},
			username:         "system:serviceaccount:user-ns:planner-serviceaccount",
			expectAllowed:    true,
		},
		{
			name:             "planner SA allowed in another namespace",
			allowedModifiers: nil,
			username:         "system:serviceaccount:other-ns:planner-serviceaccount",
			expectAllowed:    true,
		},
		{
			name:             "unauthorized SA rejected",
			allowedModifiers: []string{"system:serviceaccount:dynamo-system:dynamo-operator-controller-manager"},
			username:         "system:serviceaccount:user-ns:some-random-sa",
			expectAllowed:    false,
		},
		{
			name:             "non-SA user rejected",
			allowedModifiers: []string{"system:serviceaccount:dynamo-system:dynamo-operator-controller-manager"},
			username:         "admin@example.com",
			expectAllowed:    false,
		},
		{
			name:             "malformed SA username rejected",
			allowedModifiers: []string{"system:serviceaccount:dynamo-system:dynamo-operator-controller-manager"},
			username:         "system:serviceaccount:only-three-parts",
			expectAllowed:    false,
		},
		{
			name:             "empty allow-list still permits planner",
			allowedModifiers: nil,
			username:         "system:serviceaccount:ns:planner-serviceaccount",
			expectAllowed:    true,
		},
		{
			name:             "empty allow-list rejects operator SA",
			allowedModifiers: nil,
			username:         "system:serviceaccount:ns:dynamo-operator-controller-manager",
			expectAllowed:    false,
		},
		{
			name:             "bare SA name in allow-list does not match (must be full principal)",
			allowedModifiers: []string{"dynamo-operator-controller-manager"},
			username:         "system:serviceaccount:ns:dynamo-operator-controller-manager",
			expectAllowed:    false,
		},
		{
			name:             "multiple principals in allow-list",
			allowedModifiers: []string{"system:serviceaccount:ns:operator-cm", "system:serviceaccount:ns:external-autoscaler"},
			username:         "system:serviceaccount:ns:external-autoscaler",
			expectAllowed:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prev := allowedDGDReplicasModifiers
			t.Cleanup(func() { allowedDGDReplicasModifiers = prev })

			SetAllowedDGDReplicasModifiers(tt.allowedModifiers)

			userInfo := authenticationv1.UserInfo{Username: tt.username}
			got := CanModifyDGDReplicas(userInfo)
			if got != tt.expectAllowed {
				t.Errorf("CanModifyDGDReplicas() = %v, want %v", got, tt.expectAllowed)
			}
		})
	}
}
