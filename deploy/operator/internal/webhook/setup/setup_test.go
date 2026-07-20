/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package setup

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
)

func TestSetupAllRequiresOperatorConfiguration(t *testing.T) {
	wantErr := "operator configuration is required"
	if err := SetupAll(nil, Options{}); err == nil || err.Error() != wantErr {
		t.Fatalf("SetupAll() error = %v, want %q", err, wantErr)
	}
}

func TestSetupAllRequiresRuntimeConfiguration(t *testing.T) {
	wantErr := "runtime configuration is required"
	opts := Options{Config: &configv1alpha1.OperatorConfiguration{}}
	if err := SetupAll(nil, opts); err == nil || err.Error() != wantErr {
		t.Fatalf("SetupAll() error = %v, want %q", err, wantErr)
	}
}
