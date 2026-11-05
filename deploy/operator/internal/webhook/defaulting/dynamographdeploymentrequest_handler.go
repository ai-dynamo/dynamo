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

package defaulting

import (
	"context"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const dgdrDefaultingWebhookPath = "/mutate-nvidia-com-v1beta1-dynamographdeploymentrequest"

// DGDRDefaulter is a mutating webhook handler that fills in default values for
// native v1beta1 DynamoGraphDeploymentRequest resources.
//
// If spec.image is not set, it is derived as:
//
//	nvcr.io/nvidia/ai-dynamo/dynamo-planner:<operatorVersion>
//
// Defaulting requires a known operator version and is only supported for
// operator versions 1.0.0 and later.
type DGDRDefaulter struct {
	OperatorVersion string
	// DefaultImage is the DGDR profiler image put into spec.image when a DGDR
	// is created without one. Set it explicitly when the derived
	// dynamo-planner:<operatorVersion> tag does not exist (pre-release charts:
	// rc, nightly); empty keeps the derived default.
	DefaultImage string
	// v1beta2 configures the defaults of Requests submitted as v1beta2, which reach this
	// handler as a v1beta1 object carrying the spec.v1beta2 envelope.
	v1beta2 DGDRV1beta2Options
}

// NewDGDRDefaulter creates a new DGDRDefaulter with the given operator version
// and optional explicit default profiler image.
func NewDGDRDefaulter(operatorVersion, defaultImage string) *DGDRDefaulter {
	return &DGDRDefaulter{OperatorVersion: operatorVersion, DefaultImage: defaultImage}
}

// WithV1beta2 sets the defaults of v1beta2 Requests and returns the defaulter.
func (d *DGDRDefaulter) WithV1beta2(options DGDRV1beta2Options) *DGDRDefaulter {
	d.v1beta2 = options
	return d
}

// Default implements admission.CustomDefaulter.
// Native API defaults apply on CREATE and UPDATE. If spec.image is not set on
// CREATE, derives a default image from the operator version.
func (d *DGDRDefaulter) Default(ctx context.Context, obj runtime.Object) error {
	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK); err != nil {
		return err
	}

	dgdr, ok := obj.(*nvidiacomv1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected DynamoGraphDeploymentRequest but got %T", obj)
	}

	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		return fmt.Errorf("get admission request from context: %w", err)
	}
	requestVersion := internalwebhook.RequestedVersion(req)

	// The original request version decides what spec.v1beta2 means. A native v1beta2 Request
	// is defaulted from its carrier on CREATE and UPDATE; only CREATE
	// and an UPDATE that changes spec.hardware read the cluster.
	if requestVersion == internalwebhook.VersionV1Beta2 && dgdr.Spec.V1Beta2 != nil {
		oldSpec, err := decodeOldSpecV1beta2(req.Operation, req.OldObject.Raw)
		if err != nil {
			return err
		}
		return defaultDynamoGraphDeploymentRequestV1beta2(ctx, dgdr, req.Operation, oldSpec, d.v1beta2)
	}
	// A foreign-version view (a carrier written or updated through the wrong API version, or
	// a v1beta1 payload written through v1beta2) is read-only; validation enforces that.
	if requestVersion == internalwebhook.VersionV1Beta2 || dgdr.Spec.V1Beta2 != nil {
		return nil
	}

	defaultDynamoGraphDeploymentRequest(dgdr, dynamoGraphDeploymentRequestDefaultOptions{
		operation: req.Operation,
		image:     PlannerImageFor(d.OperatorVersion, d.DefaultImage),
	})
	return nil
}

// RegisterWithManager registers the DGDR defaulting webhook with the manager.
func (d *DGDRDefaulter) RegisterWithManager(mgr manager.Manager) error {
	defaulter := internalwebhook.NewLeaseAwareDefaulter(d, internalwebhook.GetExcludedNamespaces())
	webhook := admission.
		WithCustomDefaulter(mgr.GetScheme(), &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}, defaulter).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dgdrDefaultingWebhookPath, webhook)
	return nil
}
