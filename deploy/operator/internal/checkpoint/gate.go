/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"errors"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
)

// ValidateEnabled rejects checkpoint functionality when it is disabled in the
// operator configuration.
func ValidateEnabled(gate features.Gate) error {
	if gate.Enabled(features.Checkpoint) {
		return nil
	}
	return errors.New("checkpoint functionality is disabled in the operator configuration")
}
