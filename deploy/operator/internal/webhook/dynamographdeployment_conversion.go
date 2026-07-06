/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package webhook

import (
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// ConvertDynamoGraphDeploymentToV1Beta1 converts an admission object from the
// v1alpha1 spoke to the v1beta1 hub used by the DGD admission logic.
func ConvertDynamoGraphDeploymentToV1Beta1(src *nvidiacomv1alpha1.DynamoGraphDeployment) (*nvidiacomv1beta1.DynamoGraphDeployment, error) {
	dst := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(dst); err != nil {
		return nil, fmt.Errorf("convert v1alpha1 DynamoGraphDeployment to v1beta1: %w", err)
	}
	return dst, nil
}

// ConvertDynamoGraphDeploymentToV1Alpha1 converts a v1beta1 hub object to the
// v1alpha1 spoke expected by the legacy mutation endpoint.
func ConvertDynamoGraphDeploymentToV1Alpha1(src *nvidiacomv1beta1.DynamoGraphDeployment) (*nvidiacomv1alpha1.DynamoGraphDeployment, error) {
	dst := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	if err := dst.ConvertFrom(src); err != nil {
		return nil, fmt.Errorf("convert v1beta1 DynamoGraphDeployment to v1alpha1: %w", err)
	}
	return dst, nil
}
