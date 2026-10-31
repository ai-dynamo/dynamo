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

package validation

import (
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// The webhook boundary protects raw conversion carriers, so a missing admission request must
// fail the call instead of counting as a native write.
func TestDGDRValidatorRequiresTheAdmissionRequest(t *testing.T) {
	validator := NewDynamoGraphDeploymentRequestValidator()
	request := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}

	if _, err := validator.Validate(context.Background(), request); err == nil {
		t.Fatal("Validate must fail without an admission request in the context")
	}
	if _, err := validator.ValidateUpdate(context.Background(), request, request); err == nil {
		t.Fatal("ValidateUpdate must fail without an admission request in the context")
	}
}
