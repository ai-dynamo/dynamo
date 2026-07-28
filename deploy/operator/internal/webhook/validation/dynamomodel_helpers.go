/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// invalidDynamoModelError converts allErrs for model into an API error. model must not be nil.
func invalidDynamoModelError(model *nvidiacomv1alpha1.DynamoModel, allErrs field.ErrorList) error {
	if len(allErrs) == 0 {
		return nil
	}
	return k8serrors.NewInvalid(
		nvidiacomv1alpha1.GroupVersion.WithKind("DynamoModel").GroupKind(),
		model.Name,
		allErrs,
	)
}

func hasSupportedModelSourceScheme(uri string) bool {
	return strings.HasPrefix(uri, "s3://") ||
		strings.HasPrefix(uri, "hf://") ||
		strings.HasPrefix(uri, "file:///")
}
