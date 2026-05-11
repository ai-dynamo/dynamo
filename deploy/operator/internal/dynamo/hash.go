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

package dynamo

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// ComputeLegacyAlphaDGDWorkersSpecHash returns the v1alpha1 worker hash that a
// pre-v1beta1 controller would compute for the DGD's current spec. Conversion
// must preserve every v1alpha1 hash input shape this depends on.
func ComputeLegacyAlphaDGDWorkersSpecHash(dgd *v1beta1.DynamoGraphDeployment) (string, error) {
	alpha := &v1alpha1.DynamoGraphDeployment{}
	if err := alpha.ConvertFrom(dgd); err != nil {
		return "", err
	}
	return v1alpha1.ComputeDGDWorkersSpecHash(alpha)
}
