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
	defaultImage = "nvcr.io/nvidia/ai-dynamo/dynamo-frontend"
)

// DGDRDefaulter is a mutating webhook handler that fills in default values for
// DynamoGraphDeploymentRequest resources on CREATE.
//
// If spec.image is not set, it is derived as:
//
//	nvcr.io/nvidia/ai-dynamo/dynamo-frontend:<operatorVersion>
//
// Defaulting requires a known operator version and is only supported for
// operator versions 1.0.0 and later.
type DGDRDefaulter struct {
	OperatorVersion string
}

// NewDGDRDefaulter creates a new DGDRDefaulter with the given operator version.
func NewDGDRDefaulter(operatorVersion string) *DGDRDefaulter {
	return &DGDRDefaulter{OperatorVersion: operatorVersion}
}

// Default implements admission.CustomDefaulter.
// Only called on CREATE (the webhook is not registered for UPDATE).
// If spec.image is not set, derives a default image from the backend and operator version.
func (d *DGDRDefaulter) Default(ctx context.Context, obj runtime.Object) error {
	logger := log.FromContext(ctx).WithName(dgdrDefaultingWebhookName)

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

// defaultImageFor returns the default image, or empty string when the operator version
// is unknown (e.g. local dev builds), in which case the user must provide spec.image explicitly.
func (d *DGDRDefaulter) defaultImageFor() string {
	if d.OperatorVersion == "" || d.OperatorVersion == "unknown" {
		return ""
	}
	return fmt.Sprintf("%s:%s", defaultImage, d.OperatorVersion)
}

// RegisterWithManager registers the DGDR defaulting webhook with the manager.
func (d *DGDRDefaulter) RegisterWithManager(mgr manager.Manager) error {
	webhook := admission.
		WithCustomDefaulter(mgr.GetScheme(), &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}, d).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dgdrDefaultingWebhookPath, webhook)
	return nil
}
