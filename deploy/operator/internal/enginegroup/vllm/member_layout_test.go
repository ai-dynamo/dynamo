/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"fmt"
	"testing"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/require"
)

func TestReplicaLayoutKeepsAllocationAndNativeScaleUnitsSeparate(t *testing.T) {
	for _, width := range []int32{1, 4} {
		t.Run(fmt.Sprintf("width-%d", width), func(t *testing.T) {
			t.Log("convert three allocations to their native target without rounding or changing scale units")
			layout := ReplicaLayout{NativeMembersPerReplica: width}
			size, err := layout.NativeSize(3)
			require.NoError(t, err)
			require.Equal(t, 3*width, size)

			t.Log("assign the third allocation its complete immutable rank range")
			members, err := layout.Members(2)
			require.NoError(t, err)
			require.Len(t, members, int(width))
			require.Equal(t, domain.NativeMemberID(fmt.Sprintf("dp-%d", 2*width)), members[0])
			require.Equal(t, domain.NativeMemberID(fmt.Sprintf("dp-%d", 3*width-1)), members[len(members)-1])
		})
	}
}
