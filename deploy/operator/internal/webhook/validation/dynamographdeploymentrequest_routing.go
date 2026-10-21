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
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// requestIsNative reports whether the represented spec is native to the API version the
// client used (see internalwebhook.RequestIsNative). hasV1Beta2Carrier is whether the
// converted v1beta1 object carries spec.v1beta2. Without an admission request in the
// context (direct calls from unit tests) there is no foreign view, so the request counts as
// native.
func (v *dynamoGraphDeploymentRequestValidation) requestIsNative(hasV1Beta2Carrier bool) bool {
	req, err := admission.RequestFromContext(v.ctx)
	if err != nil {
		return true
	}
	return internalwebhook.RequestIsNative(req, hasV1Beta2Carrier)
}

// foreignCreateErrors rejects a spec that conversion alone may produce. The carrier is raw
// JSON that bypasses structural pruning; it is safe only when conversion built it from a
// native request that was validated against its own versioned schema.
func foreignCreateErrors(fldPath *field.Path, hasV1Beta2Carrier bool) field.ErrorList {
	if hasV1Beta2Carrier {
		return field.ErrorList{field.Forbidden(
			fldPath.Child("v1beta2"),
			"spec.v1beta2 is a conversion carrier that only conversion may write; create the Request through the v1beta2 API",
		)}
	}
	return field.ErrorList{field.Forbidden(
		fldPath,
		"a v1beta1 representation cannot be written through the v1beta2 API; spec.v1beta1 is a conversion carrier that only conversion may write",
	)}
}

// foreignUpdateErrors rejects a change to a spec viewed through the API version that does
// not own it. An unchanged spec never reaches this function.
func foreignUpdateErrors(fldPath *field.Path, hasV1Beta2Carrier bool) field.ErrorList {
	if hasV1Beta2Carrier {
		return field.ErrorList{field.Forbidden(
			fldPath.Child("v1beta2"),
			"spec.v1beta2 is a read-only view through this API version; update the Request through the v1beta2 API",
		)}
	}
	return field.ErrorList{field.Forbidden(
		fldPath,
		"this Request is owned by v1beta1; its spec is read-only through the v1beta2 API, update it through v1beta1",
	)}
}
