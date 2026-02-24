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

// RuntimeConfig holds runtime state that is resolved after startup (e.g., auto-detection results).
// This is separate from the static OperatorConfiguration loaded from config files.
type RuntimeConfig struct {
	// GroveEnabled is the resolved Grove availability (config override merged with auto-detection)
	GroveEnabled bool
	// LWSEnabled is the resolved LWS availability (config override merged with auto-detection)
	LWSEnabled bool
	// KaiSchedulerEnabled is the resolved Kai-scheduler availability (config override merged with auto-detection)
	KaiSchedulerEnabled bool
	// ExcludedNamespaces for cluster-wide mode namespace filtering
	ExcludedNamespaces ExcludedNamespacesInterface
}
