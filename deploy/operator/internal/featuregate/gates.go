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

package featuregate

import semver "github.com/Masterminds/semver/v3"

// Feature gates gated on the operator origin version (the operator version that
// first reconciled / created the DGD resource).

var (
	// VLLMMultiprocessing gates the use of vLLM native multiprocessing (mp)
	// instead of Ray for multi-node deployments. Enabled for DGDs originally
	// created by operator >= 1.0.0-dev.
	VLLMMultiprocessing = OperatorOriginFeatureGate{
		Name:             "VLLMMultiprocessing",
		MinOriginVersion: *semver.MustParse("1.0.0-dev"),
	}
)
