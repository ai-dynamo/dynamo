//go:build conformance

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package conformance_test

import (
	"flag"
	"strings"
	"testing"

	"sigs.k8s.io/mcs-api/conformance"
)

func TestConformance(t *testing.T) {
	contexts := strings.Split(flag.Lookup("contexts").Value.String(), ",")
	distinct := make(map[string]bool, len(contexts))
	for _, name := range contexts {
		if name = strings.TrimSpace(name); name != "" {
			distinct[name] = true
		}
	}
	if len(distinct) < 2 {
		t.Fatal("MCS conformance requires at least two distinct --contexts; use make conformance")
	}
	conformance.TestConformance(t)
}
