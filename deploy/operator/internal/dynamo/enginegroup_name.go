/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"crypto/sha256"
	"fmt"
	"strconv"
	"strings"
)

// EngineGroupNameForComponent returns the stable name of one indexed world, independent of workload revisions.
// worldIndex must be nonnegative and identifies a world, not a capacity slot within it.
func EngineGroupNameForComponent(dgdName, componentName string, worldIndex int32) string {
	name := strings.ToLower(dgdName + "-" + componentName + "-" + strconv.FormatInt(int64(worldIndex), 10))
	if len(name) <= 63 {
		return name
	}

	// Keep names usable as label values without losing uniqueness after truncation.
	digest := sha256.Sum256([]byte(name))
	return fmt.Sprintf("%s-%x", strings.TrimRight(name[:50], "-."), digest[:6])
}
