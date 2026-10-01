/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import (
	apierrors "k8s.io/apimachinery/pkg/api/errors"
)

// IgnoreIntermediateError returns err for a real (terminal) Kubernetes API error that will not
// resolve on retry — Invalid, BadRequest, or Forbidden — and nil for an intermediate
// (transient, retryable) error. It mirrors client.IgnoreNotFound, so callers can write:
//
//	if controller_common.IgnoreIntermediateError(err) != nil { /* terminal: handle */ }
//
// to act only on terminal failures while letting transient ones requeue.
func IgnoreIntermediateError(err error) error {
	if apierrors.IsInvalid(err) || apierrors.IsBadRequest(err) || apierrors.IsForbidden(err) {
		return err
	}
	return nil
}
