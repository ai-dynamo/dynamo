// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"

// IsLPX reports whether this shared spec uses the LPX integration.
func (s *DynamoComponentDeploymentSharedSpec) IsLPX() bool {
	return s.ComponentType == commonconsts.ComponentTypeLPX
}
