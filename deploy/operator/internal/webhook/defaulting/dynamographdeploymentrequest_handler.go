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

	semver "github.com/Masterminds/semver/v3"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	dgdrDefaultingWebhookName = "dynamographdeploymentrequest-defaulting-webhook"
	dgdrDefaultingWebhookPath = "/mutate-nvidia-com-v1beta1-dynamographdeploymentrequest"

	// defaultImage is the default profiler image used when spec.image is not set.
	// Default image derivation is only supported for public release versions (1.0.0+).
	//
	// Starting with Dynamo 1.1.0, the profiler's runtime dependencies
	// (kubernetes_asyncio, pmdarima, prophet, aiconfigurator, ...) ship only in the
	// dedicated dynamo-planner image, so we default to that image here. Users who
	// pin an earlier version may continue to override spec.image explicitly with
	// the frontend image they were using before.
	defaultImage = "nvcr.io/nvidia/ai-dynamo/dynamo-planner"
)

// DGDRDefaulter is a mutating webhook handler that fills in default values for
// DynamoGraphDeploymentRequest resources on CREATE.
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
}

// NewDGDRDefaulter creates a new DGDRDefaulter with the given operator version
// and optional explicit default profiler image.
func NewDGDRDefaulter(operatorVersion, defaultImage string) *DGDRDefaulter {
	return &DGDRDefaulter{OperatorVersion: operatorVersion, DefaultImage: defaultImage}
}

// Default implements admission.CustomDefaulter.
// If spec.image is not set, derives a default image from the backend and operator version.
// UPDATE requests are admitted unchanged so an omitted image is not rewritten after
// creation.
func (d *DGDRDefaulter) Default(ctx context.Context, obj runtime.Object) error {
	logger := log.FromContext(ctx).WithName(dgdrDefaultingWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK); err != nil {
		return err
	}

	dgdr, ok := obj.(*nvidiacomv1beta1.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected DynamoGraphDeploymentRequest but got %T", obj)
	}

	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		logger.Error(err, "failed to get admission request from context, skipping defaulting")
		return nil
	}

	if req.Operation == admissionv1.Create && dgdr.Spec.Image == "" {
		if img := d.defaultImageFor(); img != "" {
			dgdr.Spec.Image = img
			logger.Info("defaulted spec.image from operator version",
				"name", dgdr.Name,
				"namespace", dgdr.Namespace,
				"image", img,
			)
		}
	}

	return nil
}

// defaultImageFor returns the profiler image for spec.image: DefaultImage when
// set, else the derived image with a canonical semver tag, or an empty string
// when the operator version cannot be parsed.
func (d *DGDRDefaulter) defaultImageFor() string {
	if d.DefaultImage != "" {
		return d.DefaultImage
	}
	version, err := semver.NewVersion(d.OperatorVersion)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%s:%s", defaultImage, version.String())
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
