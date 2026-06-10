/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package common

import (
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// IsGrovePathway reports whether a DGD should use Grove based on operator
// configuration and the per-deployment opt-out annotation.
func IsGrovePathway(groveEnabled bool, annotations map[string]string) bool {
	if !groveEnabled {
		return false
	}
	return annotations == nil ||
		strings.ToLower(annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse
}
