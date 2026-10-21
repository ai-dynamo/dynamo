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

import "sigs.k8s.io/controller-runtime/pkg/webhook/admission"

const nativeCarrierVersion = "v1beta2"

// RequestedVersion is the API version the client used. With matchPolicy: Equivalent the API
// server converts the object to the version the webhook registered for; RequestKind then
// carries the original kind and Kind the converted one. RequestKind is absent when no
// conversion happened.
func RequestedVersion(req admission.Request) string {
	if req.RequestKind != nil {
		return req.RequestKind.Version
	}
	return req.Kind.Version
}

// RequestUsesV1Beta2 reports whether the client addressed the Request through v1beta2.
func RequestUsesV1Beta2(req admission.Request) bool {
	return RequestedVersion(req) == nativeCarrierVersion
}

// RequestIsNative reports whether the represented spec is native to the API version the
// client used. hasV1Beta2Carrier is whether the converted v1beta1 object carries spec.v1beta2.
// After conversion that carrier means the represented object is native v1beta2, so the
// request is native exactly when both agree. A mismatch is a foreign-version view: the
// client either wrote a conversion carrier directly or reads or updates an object through
// the API version that does not own it.
func RequestIsNative(req admission.Request, hasV1Beta2Carrier bool) bool {
	return RequestUsesV1Beta2(req) == hasV1Beta2Carrier
}
