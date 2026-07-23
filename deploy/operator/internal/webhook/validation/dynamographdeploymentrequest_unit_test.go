/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	admissionv1 "k8s.io/api/admission/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestDynamoGraphDeploymentRequestHandlerBoundaryErrorsRemainRegular(t *testing.T) {
	handler := NewDynamoGraphDeploymentRequestHandler()
	ctx := dgdAdmissionContext(admissionv1.Create, nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK)
	_, err := handler.ValidateCreate(ctx, &runtime.Unknown{})
	if err == nil || !strings.Contains(err.Error(), "expected DynamoGraphDeploymentRequest") {
		t.Fatalf("ValidateCreate() error = %v, want cast error", err)
	}
	if k8serrors.IsInvalid(err) {
		t.Fatalf("ValidateCreate() error = %v, want regular boundary error", err)
	}
}
