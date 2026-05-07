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

// ComputeLegacyAlphaDGDWorkersSpecHash returns the v1alpha1 worker hash used by
// pre-v1beta1 controllers. Converted v1alpha1 DGDs carry the exact old hash in
// an annotation; the ConvertFrom fallback is only for synthetic beta objects.
func ComputeLegacyAlphaDGDWorkersSpecHash(dgd *v1beta1.DynamoGraphDeployment) (string, error) {
	if hash := v1alpha1.GetDGDLegacyWorkerHash(dgd); hash != "" {
		return hash, nil
	}
	alpha := &v1alpha1.DynamoGraphDeployment{}
	if err := alpha.ConvertFrom(dgd); err != nil {
		return "", err
	}
	return v1alpha1.ComputeDGDWorkersSpecHash(alpha)
}

func ClearLegacyAlphaDGDWorkersSpecHash(dgd *v1beta1.DynamoGraphDeployment) {
	v1alpha1.ClearDGDLegacyWorkerHash(dgd)
}
