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

// VersionV1Beta2 is the API version whose Requests reach the webhooks as a converted v1beta1
// object carrying spec.v1beta2.
const VersionV1Beta2 = "v1beta2"

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
