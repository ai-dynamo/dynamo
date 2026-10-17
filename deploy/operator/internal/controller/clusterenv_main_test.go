//go:build clustertest && !upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	os.Exit(runClusterTests(m, nil))
}
