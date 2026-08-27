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

package v1beta2

import (
	"encoding/json"
	"fmt"
	"reflect"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"sigs.k8s.io/controller-runtime/pkg/conversion"
)

var _ conversion.Convertible = &DynamoGraphDeploymentRequest{}

// ConvertTo converts this v1beta2 spoke to the v1beta1 hub.
func (src *DynamoGraphDeploymentRequest) ConvertTo(dstRaw conversion.Hub) error {
	// Require the registered v1beta1 conversion hub.
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", dstRaw)
	}

	// Preserve metadata independently from the versioned payloads.
	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	// Unwrap a foreign v1beta1 spec or envelope the complete native v1beta2 spec.
	if src.Spec.V1Beta1 != nil {
		if hasNativeV1Beta2Spec(src.Spec) {
			return fmt.Errorf("v1beta2 spec contains both native fields and v1beta1 envelope")
		}
		if err := unmarshalEnvelope(src.Spec.V1Beta1, &dst.Spec, "spec.v1beta1"); err != nil {
			return err
		}
	} else {
		envelope, err := marshalEnvelope(src.Spec)
		if err != nil {
			return fmt.Errorf("marshal native v1beta2 spec: %w", err)
		}
		dst.Spec.V1Beta2 = envelope
	}

	// Unwrap a foreign v1beta1 status or envelope the complete native v1beta2 status.
	if src.Status.V1Beta1 != nil {
		if hasNativeV1Beta2Status(src.Status) {
			return fmt.Errorf("v1beta2 status contains both native fields and v1beta1 envelope")
		}
		if err := unmarshalEnvelope(src.Status.V1Beta1, &dst.Status, "status.v1beta1"); err != nil {
			return err
		}
	} else {
		envelope, err := marshalEnvelope(src.Status)
		if err != nil {
			return fmt.Errorf("marshal native v1beta2 status: %w", err)
		}
		dst.Status.V1Beta2 = envelope
	}

	return nil
}

// ConvertFrom converts the v1beta1 hub to this v1beta2 spoke.
func (dst *DynamoGraphDeploymentRequest) ConvertFrom(srcRaw conversion.Hub) error {
	// Require the registered v1beta1 conversion hub.
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", srcRaw)
	}

	// Preserve metadata independently from the versioned payloads.
	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()

	// Unwrap a foreign v1beta2 spec or envelope the complete native v1beta1 spec.
	if src.Spec.V1Beta2 != nil {
		if hasNativeV1Beta1Spec(src.Spec) {
			return fmt.Errorf("v1beta1 spec contains both native fields and v1beta2 envelope")
		}
		if err := unmarshalEnvelope(src.Spec.V1Beta2, &dst.Spec, "spec.v1beta2"); err != nil {
			return err
		}
	} else {
		envelope, err := marshalEnvelope(src.Spec)
		if err != nil {
			return fmt.Errorf("marshal native v1beta1 spec: %w", err)
		}
		dst.Spec.V1Beta1 = envelope
	}

	// Unwrap a foreign v1beta2 status or envelope the complete native v1beta1 status.
	if src.Status.V1Beta2 != nil {
		if hasNativeV1Beta1Status(src.Status) {
			return fmt.Errorf("v1beta1 status contains both native fields and v1beta2 envelope")
		}
		if err := unmarshalEnvelope(src.Status.V1Beta2, &dst.Status, "status.v1beta2"); err != nil {
			return err
		}
	} else {
		envelope, err := marshalEnvelope(src.Status)
		if err != nil {
			return fmt.Errorf("marshal native v1beta1 status: %w", err)
		}
		dst.Status.V1Beta1 = envelope
	}

	return nil
}

func marshalEnvelope(value any) (*apiextensionsv1.JSON, error) {
	// Encode the complete native payload as a public JSON envelope.
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return &apiextensionsv1.JSON{Raw: raw}, nil
}

func unmarshalEnvelope(envelope *apiextensionsv1.JSON, dst any, field string) error {
	// Reject envelopes without a JSON payload.
	if len(envelope.Raw) == 0 {
		return fmt.Errorf("%s envelope is empty", field)
	}

	// Decode the complete foreign payload into its native destination type.
	if err := json.Unmarshal(envelope.Raw, dst); err != nil {
		return fmt.Errorf("unmarshal %s envelope: %w", field, err)
	}
	return nil
}

func hasNativeV1Beta2Spec(spec DynamoGraphDeploymentRequestSpec) bool {
	spec.V1Beta1 = nil
	return !reflect.DeepEqual(spec, DynamoGraphDeploymentRequestSpec{})
}

func hasNativeV1Beta2Status(status DynamoGraphDeploymentRequestStatus) bool {
	status.V1Beta1 = nil
	return !reflect.DeepEqual(status, DynamoGraphDeploymentRequestStatus{})
}

func hasNativeV1Beta1Spec(spec v1beta1.DynamoGraphDeploymentRequestSpec) bool {
	spec.V1Beta2 = nil
	return !reflect.DeepEqual(spec, v1beta1.DynamoGraphDeploymentRequestSpec{})
}

func hasNativeV1Beta1Status(status v1beta1.DynamoGraphDeploymentRequestStatus) bool {
	status.V1Beta2 = nil
	return !reflect.DeepEqual(status, v1beta1.DynamoGraphDeploymentRequestStatus{})
}
