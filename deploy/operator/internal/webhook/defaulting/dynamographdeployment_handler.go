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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	dgdDefaultingWebhookName = "dynamographdeployment-defaulting-webhook"
	dgdDefaultingWebhookPath = "/mutate-nvidia-com-v1alpha1-dynamographdeployment"
)

// DGDDefaulter is a mutating webhook handler that stamps DynamoGraphDeployments
// with the operator version on CREATE. This provides a general-purpose mechanism
// for version-gated behavior changes in the controller.
type DGDDefaulter struct {
	OperatorVersion string
}

// NewDGDDefaulter creates a new DGDDefaulter with the given operator version.
func NewDGDDefaulter(operatorVersion string) *DGDDefaulter {
	return &DGDDefaulter{
		OperatorVersion: operatorVersion,
	}
}

// Default implements admission.CustomDefaulter.
// On CREATE: stamps nvidia.com/dynamo-operator-origin-version with the operator version.
// On UPDATE: does nothing -- the origin version is immutable once set.
func (d *DGDDefaulter) Default(ctx context.Context, obj runtime.Object) error {
	logger := log.FromContext(ctx).WithName(dgdDefaultingWebhookName)

	dgd, ok := obj.(*nvidiacomv1alpha1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected DynamoGraphDeployment but got %T", obj)
	}

	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		logger.Error(err, "failed to get admission request from context, skipping defaulting")
		return nil
	}

	if req.Operation == admissionv1.Create {
		if dgd.Annotations == nil {
			dgd.Annotations = make(map[string]string)
		}
		// Stamp operator version on creation (don't overwrite if already set)
		if _, exists := dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]; !exists {
			dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = d.OperatorVersion
			logger.Info("stamped operator origin version on DGD",
				"name", dgd.Name,
				"namespace", dgd.Namespace,
				"version", d.OperatorVersion)
		}
	}

	return nil
}

// RegisterWithManager registers the defaulting webhook with the manager.
func (d *DGDDefaulter) RegisterWithManager(mgr manager.Manager) error {
	webhook := admission.
		WithCustomDefaulter(mgr.GetScheme(), &nvidiacomv1alpha1.DynamoGraphDeployment{}, d).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dgdDefaultingWebhookPath, webhook)
	return nil
}
