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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/conversion"
)

var _ conversion.Convertible = &DynamoGraphDeploymentRequest{}

// nativePayload excludes public outer types from conversion envelopes at compile time.
type nativePayload interface {
	DynamoGraphDeploymentRequestNativeSpec |
		DynamoGraphDeploymentRequestNativeStatus |
		v1beta1.DynamoGraphDeploymentRequestNativeSpec |
		v1beta1.DynamoGraphDeploymentRequestNativeStatus
}

// ConvertTo converts this v1beta2 spoke to the v1beta1 hub.
func (src *DynamoGraphDeploymentRequest) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", dstRaw)
	}

	return convertRequestPayloads(
		&src.ObjectMeta, &dst.ObjectMeta,
		src.Spec.DynamoGraphDeploymentRequestNativeSpec, src.Spec.V1Beta1,
		&dst.Spec.DynamoGraphDeploymentRequestNativeSpec,
		func(envelope *apiextensionsv1.JSON) { dst.Spec.V1Beta2 = envelope },
		src.Status.DynamoGraphDeploymentRequestNativeStatus, src.Status.V1Beta1,
		&dst.Status.DynamoGraphDeploymentRequestNativeStatus,
		func(envelope *apiextensionsv1.JSON) { dst.Status.V1Beta2 = envelope },
		"v1beta2", "v1beta1",
	)
}

// ConvertFrom converts the v1beta1 hub to this v1beta2 spoke.
func (dst *DynamoGraphDeploymentRequest) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoGraphDeploymentRequest but got %T", srcRaw)
	}

	return convertRequestPayloads(
		&src.ObjectMeta, &dst.ObjectMeta,
		src.Spec.DynamoGraphDeploymentRequestNativeSpec, src.Spec.V1Beta2,
		&dst.Spec.DynamoGraphDeploymentRequestNativeSpec,
		func(envelope *apiextensionsv1.JSON) { dst.Spec.V1Beta1 = envelope },
		src.Status.DynamoGraphDeploymentRequestNativeStatus, src.Status.V1Beta2,
		&dst.Status.DynamoGraphDeploymentRequestNativeStatus,
		func(envelope *apiextensionsv1.JSON) { dst.Status.V1Beta1 = envelope },
		"v1beta1", "v1beta2",
	)
}

func convertRequestPayloads[S, D, SS, DS nativePayload](
	srcMeta, dstMeta *metav1.ObjectMeta,
	srcSpec S,
	foreignSpec *apiextensionsv1.JSON,
	dstSpec *D,
	setSpecEnvelope func(*apiextensionsv1.JSON),
	srcStatus SS,
	foreignStatus *apiextensionsv1.JSON,
	dstStatus *DS,
	setStatusEnvelope func(*apiextensionsv1.JSON),
	srcVersion, foreignVersion string,
) error {
	*dstMeta = *srcMeta.DeepCopy()
	if err := convertPayload(
		srcSpec, foreignSpec, dstSpec, setSpecEnvelope,
		srcVersion, foreignVersion, "spec",
	); err != nil {
		return err
	}

	// Empty status has no representation discriminator. Keep it empty instead
	// of manufacturing an envelope after the status subresource resets it.
	if foreignStatus == nil && !hasNativeFields(srcStatus) {
		return nil
	}

	return convertPayload(
		srcStatus, foreignStatus, dstStatus, setStatusEnvelope,
		srcVersion, foreignVersion, "status",
	)
}

func convertPayload[S, D nativePayload](
	src S,
	foreignEnvelope *apiextensionsv1.JSON,
	dst *D,
	setNativeEnvelope func(*apiextensionsv1.JSON),
	srcVersion, foreignVersion, field string,
) error {
	if foreignEnvelope != nil {
		if hasNativeFields(src) {
			return fmt.Errorf(
				"%s %s contains both native fields and %s envelope",
				srcVersion, field, foreignVersion,
			)
		}
		return unmarshalEnvelope(
			foreignEnvelope,
			dst,
			fmt.Sprintf("%s.%s", field, foreignVersion),
		)
	}

	envelope, err := marshalEnvelope(src)
	if err != nil {
		return fmt.Errorf("marshal native %s %s: %w", srcVersion, field, err)
	}
	setNativeEnvelope(envelope)
	return nil
}

func marshalEnvelope[T nativePayload](value T) (*apiextensionsv1.JSON, error) {
	// Encode the complete native payload as a public JSON envelope.
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return &apiextensionsv1.JSON{Raw: raw}, nil
}

func unmarshalEnvelope[T nativePayload](envelope *apiextensionsv1.JSON, dst *T, field string) error {
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

func hasNativeFields[T nativePayload](value T) bool {
	var zero T
	return !reflect.DeepEqual(value, zero)
}

func hasNativeV1Beta2Spec(spec DynamoGraphDeploymentRequestNativeSpec) bool {
	return hasNativeFields(spec)
}

func hasNativeV1Beta2Status(status DynamoGraphDeploymentRequestNativeStatus) bool {
	return hasNativeFields(status)
}

func hasNativeV1Beta1Spec(spec v1beta1.DynamoGraphDeploymentRequestNativeSpec) bool {
	return hasNativeFields(spec)
}

func hasNativeV1Beta1Status(status v1beta1.DynamoGraphDeploymentRequestNativeStatus) bool {
	return hasNativeFields(status)
}
