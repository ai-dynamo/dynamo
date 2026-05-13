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
