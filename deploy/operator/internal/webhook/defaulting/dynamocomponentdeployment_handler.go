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
	dcdDefaultingWebhookName = "dynamocomponentdeployment-defaulting-webhook"
	dcdDefaultingWebhookPath = "/mutate-nvidia-com-v1beta1-dynamocomponentdeployment"
)

// DCDDefaulter is a mutating webhook handler for v1beta1 DynamoComponentDeployment.
type DCDDefaulter struct{}

// NewDCDDefaulter creates a new DCDDefaulter.
func NewDCDDefaulter() *DCDDefaulter {
	return &DCDDefaulter{}
}

// Default implements admission.CustomDefaulter.
// On CREATE, standalone v1beta1 DCDs default spec.name from metadata.name.
func (d *DCDDefaulter) Default(ctx context.Context, obj runtime.Object) error {
	logger := log.FromContext(ctx).WithName(dcdDefaultingWebhookName)

	dcd, ok := obj.(*nvidiacomv1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected DynamoComponentDeployment but got %T", obj)
	}

	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		logger.Error(err, "failed to get admission request from context, skipping defaulting")
		return nil
	}

	if req.Operation == admissionv1.Create && dcd.Spec.ComponentName == "" && dcd.Name != "" {
		dcd.Spec.ComponentName = dcd.Name
		logger.Info("defaulted spec.name from metadata.name",
			"name", dcd.Name,
			"namespace", dcd.Namespace,
		)
	}

	return nil
}

// RegisterWithManager registers the DCD defaulting webhook with the manager.
func (d *DCDDefaulter) RegisterWithManager(mgr manager.Manager) error {
	webhook := admission.
		WithCustomDefaulter(mgr.GetScheme(), &nvidiacomv1beta1.DynamoComponentDeployment{}, d).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dcdDefaultingWebhookPath, webhook)
	return nil
}
