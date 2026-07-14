/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller_common

import "github.com/ai-dynamo/dynamo/deploy/operator/internal/features"

// RuntimeConfig holds runtime state that is resolved after startup (e.g., auto-detection results).
// This is separate from the static OperatorConfiguration loaded from config files.
type RuntimeConfig struct {
	// Gate contains the resolved operator features.
	Gate features.Gate
	// ExcludedNamespaces for cluster-wide mode namespace filtering
	ExcludedNamespaces ExcludedNamespacesInterface
}

// Enabled reports whether a resolved operator feature is enabled.
func (r *RuntimeConfig) Enabled(name features.Name) bool {
	if r == nil || r.Gate == nil {
		return features.Defaults().Enabled(name)
	}
	return r.Gate.Enabled(name)
}
