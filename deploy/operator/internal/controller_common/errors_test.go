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
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestIgnoreIntermediateError(t *testing.T) {
	gr := schema.GroupResource{Group: "nvidia.com", Resource: "podsnapshots"}

	terminal := map[string]error{
		"invalid":    apierrors.NewInvalid(schema.GroupKind{Group: "nvidia.com", Kind: "PodSnapshot"}, "x", nil),
		"badRequest": apierrors.NewBadRequest("bad"),
		"forbidden":  apierrors.NewForbidden(gr, "x", errors.New("not owned")),
	}
	for name, err := range terminal {
		t.Run("terminal/"+name, func(t *testing.T) {
			assert.Error(t, IgnoreIntermediateError(err), "real errors must be returned")
		})
	}

	intermediate := map[string]error{
		"conflict": apierrors.NewConflict(gr, "x", errors.New("conflict")),
		"timeout":  apierrors.NewServerTimeout(gr, "create", 1),
		"plain":    errors.New("network blip"),
		"nil":      nil,
	}
	for name, err := range intermediate {
		t.Run("intermediate/"+name, func(t *testing.T) {
			assert.NoError(t, IgnoreIntermediateError(err), "intermediate errors must be ignored")
		})
	}
}
