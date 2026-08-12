/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// invalidDynamoCheckpointError converts allErrs for checkpoint into an API error.
// checkpoint must not be nil.
func invalidDynamoCheckpointError(
	checkpoint *nvidiacomv1alpha1.DynamoCheckpoint,
	allErrs field.ErrorList,
) error {
	if len(allErrs) == 0 {
		return nil
	}
	return k8serrors.NewInvalid(
		nvidiacomv1alpha1.GroupVersion.WithKind("DynamoCheckpoint").GroupKind(),
		checkpoint.Name,
		allErrs,
	)
}

func containerIndexByName(containers []corev1.Container, name string) int {
	for i := range containers {
		if containers[i].Name == name {
			return i
		}
	}
	return -1
}
