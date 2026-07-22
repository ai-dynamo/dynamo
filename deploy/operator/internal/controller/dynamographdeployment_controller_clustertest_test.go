//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"
)

func TestClusterDynamoGraphDeploymentManifests(t *testing.T) {
	clusterTestRunManifestScenarios(t, "testdata/dgd", clusterTestDGDChain)
}
