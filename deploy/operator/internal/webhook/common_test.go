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
			allowedModifiers: []string{"dynamo-platform-dynamo-operator-controller-manager", "planner-serviceaccount"},
			username:         "system:serviceaccount:dynamo-system:dynamo-platform-dynamo-operator-controller-manager",
			expectAllowed:    true,
		},
		{
			name:             "operator SA with collapsed Helm release (dynamo-operator) — the bug scenario",
			allowedModifiers: []string{"dynamo-operator-controller-manager", "planner-serviceaccount"},
			username:         "system:serviceaccount:dynamo-system:dynamo-operator-controller-manager",
			expectAllowed:    true,
		},
		{
			name:             "operator SA with custom Helm release name",
			allowedModifiers: []string{"my-release-controller-manager", "planner-serviceaccount"},
			username:         "system:serviceaccount:custom-ns:my-release-controller-manager",
			expectAllowed:    true,
		},
		{
			name:             "planner SA allowed via config",
			allowedModifiers: []string{"dynamo-operator-controller-manager", "planner-serviceaccount"},
			username:         "system:serviceaccount:user-ns:planner-serviceaccount",
			expectAllowed:    true,
		},
		{
			name:             "planner SA in different namespace",
			allowedModifiers: []string{"dynamo-operator-controller-manager", "planner-serviceaccount"},
			username:         "system:serviceaccount:other-ns:planner-serviceaccount",
			expectAllowed:    true,
		},
		{
			name:             "unauthorized SA rejected",
			allowedModifiers: []string{"dynamo-operator-controller-manager", "planner-serviceaccount"},
			username:         "system:serviceaccount:user-ns:some-random-sa",
			expectAllowed:    false,
		},
		{
			name:             "non-SA user rejected",
			allowedModifiers: []string{"dynamo-operator-controller-manager", "planner-serviceaccount"},
			username:         "admin@example.com",
			expectAllowed:    false,
		},
		{
			name:             "malformed SA username rejected",
			allowedModifiers: []string{"dynamo-operator-controller-manager"},
			username:         "system:serviceaccount:only-three-parts",
			expectAllowed:    false,
		},
		{
			name:             "empty allow-list rejects all SAs",
			allowedModifiers: nil,
			username:         "system:serviceaccount:ns:planner-serviceaccount",
			expectAllowed:    false,
		},
		{
			name:             "multiple allowed SAs",
			allowedModifiers: []string{"operator-cm", "external-autoscaler"},
			username:         "system:serviceaccount:ns:external-autoscaler",
			expectAllowed:    true,
		},
		{
			name:             "partial name match is not allowed (substring)",
			allowedModifiers: []string{"controller-manager"},
			username:         "system:serviceaccount:ns:dynamo-operator-controller-manager",
			expectAllowed:    false,
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
