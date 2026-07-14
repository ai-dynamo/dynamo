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
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	dgdDefaultingWebhookName         = "dynamographdeployment-defaulting-webhook"
	dgdV1Alpha1DefaultingWebhookPath = "/mutate-nvidia-com-v1alpha1-dynamographdeployment"
	dgdV1Beta1DefaultingWebhookPath  = "/mutate/nvidia.com/v1beta1/dynamographdeployments"
)

// DGDDefaulter is a mutating webhook handler that stamps DynamoGraphDeployments
// with the operator version on CREATE. This provides a general-purpose mechanism
// for version-gated behavior changes in the controller.
type DGDDefaulter struct {
	OperatorVersion string
}

// dgdV1Alpha1Defaulter keeps the previous endpoint available during the
// v1alpha1-to-v1beta1 admission migration. It applies v1beta1 defaulting and
// converts the result back to the object version used by the legacy endpoint.
type dgdV1Alpha1Defaulter struct {
	defaulter *DGDDefaulter
}

// NewDGDDefaulter creates a new DGDDefaulter with the given operator version.
func NewDGDDefaulter(operatorVersion string) *DGDDefaulter {
	return &DGDDefaulter{
		OperatorVersion: operatorVersion,
	}
}

// Default implements admission.CustomDefaulter.
// On every operation: defaults nil Replicas to 1 for all components.
// On every Grove-pathway operation: defaults nil MinAvailable to 1. Scaling to
// replicas=0 does not rewrite MinAvailable; it remains the component's
// configured minimum viable unit.
// On CREATE: stamps nvidia.com/dynamo-operator-origin-version with the operator version.
// On UPDATE/DELETE: the origin version annotation is immutable once set.
func (d *DGDDefaulter) Default(ctx context.Context, obj runtime.Object) error {
	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentGVK); err != nil {
		return err
	}

	dgd, ok := obj.(*nvidiacomv1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected DynamoGraphDeployment but got %T", obj)
	}
	return d.defaultV1Beta1(ctx, dgd)
}

func (d *DGDDefaulter) defaultV1Beta1(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx).WithName(dgdDefaultingWebhookName)

	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		logger.Error(err, "failed to get admission request from context, skipping defaulting")
		return nil
	}

	// Default nil replicas to 1 for all components. The Replicas field is
	// *int32 with omitempty, so users can legally omit it. Without this
	// default the controller panics on a nil pointer dereference in
	// expandRolesForComponent(). Apply on every operation so that components
	// added via UPDATE also get the default.
	grovePathway := d.isGrovePathway(ctx, dgd)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.Replicas == nil {
			component.Replicas = ptr.To(int32(1))
		}
		if grovePathway && component.MinAvailable == nil {
			component.MinAvailable = ptr.To(int32(1))
		}
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

func (d *DGDDefaulter) isGrovePathway(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	return features.GateFromContext(ctx).Enabled(features.Grove) && (dgd.Annotations == nil ||
		strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse)
}

// RegisterWithManager registers the defaulting webhook with the manager.
func (d *DGDDefaulter) RegisterWithManager(mgr manager.Manager, gate features.Gate) error {
	betaDefaulter := internalwebhook.DefaulterWithGate(d, gate)
	betaDefaulter = internalwebhook.NewLeaseAwareDefaulter(betaDefaulter, internalwebhook.GetExcludedNamespaces())
	betaWebhook := admission.
		WithCustomDefaulter(mgr.GetScheme(), &nvidiacomv1beta1.DynamoGraphDeployment{}, betaDefaulter).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dgdV1Beta1DefaultingWebhookPath, betaWebhook)

	// TODO(1.5): Remove the v1alpha1 endpoint and defaulter after 1.3 is no longer
	// a supported upgrade or rollback target.
	alphaDefaulter := &dgdV1Alpha1Defaulter{defaulter: d}
	alphaDefaulterWithGate := internalwebhook.DefaulterWithGate(alphaDefaulter, gate)
	alphaDefaulterWithLease := internalwebhook.NewLeaseAwareDefaulter(alphaDefaulterWithGate, internalwebhook.GetExcludedNamespaces())
	alphaWebhook := admission.
		WithCustomDefaulter(mgr.GetScheme(), &nvidiacomv1alpha1.DynamoGraphDeployment{}, alphaDefaulterWithLease).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dgdV1Alpha1DefaultingWebhookPath, alphaWebhook)
	return nil
}

func (d *dgdV1Alpha1Defaulter) Default(ctx context.Context, obj runtime.Object) error {
	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1alpha1.DynamoGraphDeploymentGVK); err != nil {
		return err
	}

	alpha, ok := obj.(*nvidiacomv1alpha1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected DynamoGraphDeployment but got %T", obj)
	}

	beta, err := internalwebhook.ConvertDynamoGraphDeploymentToV1Beta1(alpha)
	if err != nil {
		return err
	}
	if err := d.defaulter.defaultV1Beta1(ctx, beta); err != nil {
		return err
	}

	converted, err := internalwebhook.ConvertDynamoGraphDeploymentToV1Alpha1(beta)
	if err != nil {
		return err
	}
	converted.TypeMeta = alpha.TypeMeta
	*alpha = *converted
	return nil
}
