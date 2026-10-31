/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"encoding/json"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// decodeDynamoGraphDeploymentRequestSpecV1beta2 unwraps the public conversion carrier.
// raw must not be nil; callers report decode errors at the carrier's field path.
func decodeDynamoGraphDeploymentRequestSpecV1beta2(raw *apiextensionsv1.JSON) (*nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec, error) {
	var spec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(raw.Raw, &spec); err != nil {
		return nil, err
	}
	return &spec, nil
}

// invalidDynamoGraphDeploymentRequestError converts allErrs for request into an API error.
// request must not be nil.
func invalidDynamoGraphDeploymentRequestError(
	request *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	allErrs field.ErrorList,
) error {
	if len(allErrs) == 0 {
		return nil
	}
	return k8serrors.NewInvalid(nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK.GroupKind(), request.Name, allErrs)
}

func hasManualDGDRHardware(hardware *nvidiacomv1beta1.HardwareSpec) bool {
	return hardware != nil &&
		(hardware.GPUSKU != "" || hardware.VRAMMB != nil || hardware.NumGPUsPerNode != nil)
}

func isImmutableDGDRPhase(phase nvidiacomv1beta1.DGDRPhase) bool {
	switch phase {
	case nvidiacomv1beta1.DGDRPhaseProfiling,
		nvidiacomv1beta1.DGDRPhaseDeploying,
		nvidiacomv1beta1.DGDRPhaseReady,
		nvidiacomv1beta1.DGDRPhaseDeployed:
		return true
	default:
		return false
	}
}
