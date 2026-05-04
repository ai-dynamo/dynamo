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

// Conversion scaffolding between v1alpha1 and v1beta1 DynamoGraphDeployment.
//
// This file establishes v1alpha1 as a spoke in the hub-and-spoke conversion
// model, with v1beta1 as the hub. The actual field-by-field conversion logic
// is intentionally not implemented in this MR; the real mapping lands in a
// follow-up once the v1beta1 controller is ready. Both directions currently
// return an error so that any accidental invocation fails loudly.
//
// The conversion functions are still required for v1alpha1 to satisfy the
// `conversion.Convertible` interface; without them the scheme would not compile
// a multi-version CRD.
//
// While v1beta1 is marked `+kubebuilder:unservedversion`, the API server will
// never invoke conversion -- every request resolves to the served v1alpha1
// version. These stubs exist to make the type graph complete and to fail fast
// if someone wires up conversion prematurely.

package v1alpha1

import (
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// errDGDConversionNotImplemented is returned by the v1alpha1 <-> v1beta1 DGD
// conversion stubs until real mapping logic is added.
var errDGDConversionNotImplemented = fmt.Errorf(
	"DynamoGraphDeployment v1alpha1 <-> v1beta1 conversion is not yet implemented; " +
		"v1beta1 is marked unserved, use v1alpha1")

// ConvertTo converts this DynamoGraphDeployment (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoGraphDeployment) ConvertTo(dstRaw conversion.Hub) error {
	if _, ok := dstRaw.(*v1beta1.DynamoGraphDeployment); !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", dstRaw)
	}
	return errDGDConversionNotImplemented
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoGraphDeployment (v1alpha1).
func (dst *DynamoGraphDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	if _, ok := srcRaw.(*v1beta1.DynamoGraphDeployment); !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeployment but got %T", srcRaw)
	}
	return errDGDConversionNotImplemented
}
