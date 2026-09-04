/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDGDCheckpointID(t *testing.T) {
	id := DGDCheckpointID("ns", "dgd", "uid-1", "worker", "hash-1")
	assert.Regexp(t, "^[0-9a-f]{32}$", id)
	assert.Equal(t, id, DGDCheckpointID("ns", "dgd", "uid-1", "worker", "hash-1"))

	assert.NotEqual(t, id, DGDCheckpointID("ns", "dgd", "uid-2", "worker", "hash-1"), "DGD UID must prevent cross-DGD reuse")
	assert.NotEqual(t, id, DGDCheckpointID("ns", "dgd", "uid-1", "worker", "hash-2"), "worker hash must isolate worker generations")
	assert.NotEqual(t, id, DGDCheckpointID("ns", "dgd", "uid-1", "prefill", "hash-1"), "component name must isolate components")
}
