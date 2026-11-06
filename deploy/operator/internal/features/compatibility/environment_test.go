/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package compatibility

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
)

func TestEnvironmentOriginBoundary(t *testing.T) {
	for _, test := range []struct {
		origin  string
		enabled bool
	}{
		{origin: ""}, {origin: "invalid"}, {origin: "1.7.99"}, {origin: "1.8.0-rc.1"},
		{origin: "1.8.0", enabled: true}, {origin: "v1.8", enabled: true},
		{origin: "1.8.0+build", enabled: true}, {origin: "1.8.1-dev", enabled: true},
	} {
		t.Run(test.origin, func(t *testing.T) {
			t.Log("Enable semantic ordering only when the stored origin meets 1.8.0")
			require.Equal(t, test.enabled, OrderedEnvironmentVariables.Enabled(map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: test.origin}))
		})
	}
}
