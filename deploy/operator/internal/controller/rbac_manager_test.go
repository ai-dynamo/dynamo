/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import "context"

// MockRBACManager delegates RBAC setup when a test supplies an implementation.
type MockRBACManager struct {
	EnsureServiceAccountWithRBACFunc func(ctx context.Context, targetNamespace, serviceAccountName, clusterRoleName string) error
}

func (m *MockRBACManager) EnsureServiceAccountWithRBAC(ctx context.Context, targetNamespace, serviceAccountName, clusterRoleName string) error {
	if m.EnsureServiceAccountWithRBACFunc != nil {
		return m.EnsureServiceAccountWithRBACFunc(ctx, targetNamespace, serviceAccountName, clusterRoleName)
	}
	return nil
}
