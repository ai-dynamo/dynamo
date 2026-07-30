/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package compatibility

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

func TestCanaryHealthCheckGateThresholds(t *testing.T) {
	originBelow := map[string]string{
		consts.KubeAnnotationDynamoOperatorOriginVersion: "1.3.0",
	}
	originAtThreshold := map[string]string{
		consts.KubeAnnotationDynamoOperatorOriginVersion: "1.4.0",
	}
	if CanaryHealthChecksOrigin.Enabled(originBelow) {
		t.Fatal("origin gate enabled below 1.4.0")
	}
	if !CanaryHealthChecksOrigin.Enabled(originAtThreshold) {
		t.Fatal("origin gate disabled at 1.4.0")
	}

	runtimeBelow := runtimeversion.Version{Major: 1, Minor: 3, Patch: 0}
	runtimeAtThreshold := runtimeversion.Version{Major: 1, Minor: 4, Patch: 0}
	if CanaryHealthChecksRuntime.Enabled(&runtimeBelow) {
		t.Fatal("runtime gate enabled below 1.4.0")
	}
	if !CanaryHealthChecksRuntime.Enabled(&runtimeAtThreshold) {
		t.Fatal("runtime gate disabled at 1.4.0")
	}
}
