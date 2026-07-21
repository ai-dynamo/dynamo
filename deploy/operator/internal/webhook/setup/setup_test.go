/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package setup

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
)

func TestSetupRequiresOperatorConfiguration(t *testing.T) {
	wantErr := "operator configuration is required"
	if err := Setup(nil, Options{}); err == nil || err.Error() != wantErr {
		t.Fatalf("Setup() error = %v, want %q", err, wantErr)
	}
}

func TestSetupRequiresRuntimeConfiguration(t *testing.T) {
	wantErr := "runtime configuration is required"
	opts := Options{Config: &configv1alpha1.OperatorConfiguration{}}
	if err := Setup(nil, opts); err == nil || err.Error() != wantErr {
		t.Fatalf("Setup() error = %v, want %q", err, wantErr)
	}
}
