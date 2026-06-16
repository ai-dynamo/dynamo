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
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// ValidateAdmissionGVK verifies that the API server dispatched the admission
// request to the handler version the operator supports. This catches bad
// webhook registration or unexpected API server dispatch, but it cannot prove
// that CRD schema conversion rewrote the object body.
func ValidateAdmissionGVK(ctx context.Context, expected schema.GroupVersionKind) error {
	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		return fmt.Errorf("admission request missing from context: %w", err)
	}

	got := schema.GroupVersionKind{
		Group:   req.Kind.Group,
		Version: req.Kind.Version,
		Kind:    req.Kind.Kind,
	}
	if got == expected {
		return nil
	}

	return fmt.Errorf("admission requires %s, got %s; check webhook and CRD conversion configuration", expected, got)
}
