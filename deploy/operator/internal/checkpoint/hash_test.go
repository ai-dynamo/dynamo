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
	id := DGDCheckpointID("ns", "dgd", "uid-1", "worker", "worker-hash-1", "compatibility-hash-1", "v2")
	assert.Regexp(t, "^[0-9a-f]{32}$", id)
	assert.Equal(t, id, DGDCheckpointID("ns", "dgd", "uid-1", "worker", "worker-hash-1", "compatibility-hash-1", "v2"))

	assert.NotEqual(t, id, DGDCheckpointID("ns", "dgd", "uid-2", "worker", "worker-hash-1", "compatibility-hash-1", "v2"), "DGD UID must prevent cross-DGD reuse")
	assert.NotEqual(t, id, DGDCheckpointID("ns", "dgd", "uid-1", "worker", "worker-hash-2", "compatibility-hash-1", "v2"), "worker hash must isolate worker generations")
	assert.NotEqual(t, id, DGDCheckpointID("ns", "dgd", "uid-1", "worker", "worker-hash-1", "compatibility-hash-2", "v2"), "compatibility hash must isolate rendered capture contracts")
	assert.NotEqual(t, id, DGDCheckpointID("ns", "dgd", "uid-1", "prefill", "worker-hash-1", "compatibility-hash-1", "v2"), "component name must isolate components")
	assert.NotEqual(t, id, DGDCheckpointID("ns", "dgd", "uid-1", "worker", "worker-hash-1", "compatibility-hash-1", "v1"), "compatibility version must isolate contract generations")
}
