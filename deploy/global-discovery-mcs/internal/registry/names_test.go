// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package registry

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsDNSLabel(t *testing.T) {
	for _, ok := range []string{"a", "use1-a", "0", strings.Repeat("a", 63)} {
		assert.True(t, IsDNSLabel(ok), ok)
	}
	for _, bad := range []string{"", "-a", "a-", "A", "a.b", "a_b", strings.Repeat("a", 64)} {
		assert.False(t, IsDNSLabel(bad), bad)
	}
}
