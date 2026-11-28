// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package version

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestKeysAndValues(t *testing.T) {
	kv := KeysAndValues()
	assert.Len(t, kv, 8)
	assert.Equal(t, []any{"service", Service, "version", Version, "gitHash", GitHash, "releaseTag", ReleaseTag}, kv)
}

func TestString(t *testing.T) {
	oldVersion, oldHash := Version, GitHash
	t.Cleanup(func() { Version, GitHash = oldVersion, oldHash })

	Version, GitHash = "1.2.3", "abc123"
	assert.Equal(t, "1.2.3 (abc123)", String())
}
