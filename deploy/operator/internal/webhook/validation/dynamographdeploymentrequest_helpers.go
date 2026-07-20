/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

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
		nvidiacomv1beta1.DGDRPhaseDeployed:
		return true
	default:
		return false
	}
}
