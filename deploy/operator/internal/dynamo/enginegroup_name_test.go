/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEngineGroupNameForComponent(t *testing.T) {
	cases := []struct {
		name string
		dgd  string
	}{
		{name: "short name", dgd: "graph"},
		{name: "truncated name", dgd: strings.Repeat("a", 100)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("derive stable names from the component and explicit world index")
			name := EngineGroupNameForComponent(tc.dgd, "Worker", 0)
			assert.LessOrEqual(t, len(name), 63)
			assert.Equal(t, name, EngineGroupNameForComponent(tc.dgd, "Worker", 0))
			assert.NotEqual(t, name, EngineGroupNameForComponent(tc.dgd, "Decode", 0))
			assert.NotEqual(t, name, EngineGroupNameForComponent(tc.dgd, "Worker", 1))
			assert.NotEqual(t, name, EngineGroupNameForComponent(tc.dgd, "Worker", 12))
		})
	}

	t.Log("preserve the existing world-zero name without treating every world as zero")
	assert.Equal(t, "graph-worker-0", EngineGroupNameForComponent("graph", "Worker", 0))
	assert.Equal(t, "graph-worker-12", EngineGroupNameForComponent("graph", "Worker", 12))
}
