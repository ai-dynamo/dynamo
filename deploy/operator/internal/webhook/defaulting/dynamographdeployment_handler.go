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
	dgdDefaultingWebhookName = "dynamographdeployment-defaulting-webhook"
	dgdDefaultingWebhookPath = "/mutate/nvidia.com/v1beta1/dynamographdeployments"
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
// On every operation: defaults nil Replicas to 1 for all components.
// On CREATE: sets the controller-owned workload provider from routing intent before provider-specific defaults.
// Existing unannotated DGDs remain unselected for controller-side workload adoption.
// On the Grove pathway: defaults nil MinAvailable to 1. Scaling to replicas=0
// does not rewrite MinAvailable; it remains the component's configured minimum viable unit.
// On CREATE: stamps nvidia.com/dynamo-operator-origin-version with the operator version.
// On UPDATE/DELETE: the origin version annotation is immutable once set.
func (d *DGDDefaulter) Default(ctx context.Context, obj runtime.Object) error {
	logger := log.FromContext(ctx).WithName(dgdDefaultingWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta1.DynamoGraphDeploymentGVK); err != nil {
		return err
	}

	dgd, ok := obj.(*nvidiacomv1beta1.DynamoGraphDeployment)
	if !ok {
		return fmt.Errorf("expected DynamoGraphDeployment but got %T", obj)
	}

	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		logger.Error(err, "failed to get admission request from context, skipping defaulting")
		return nil
	}

	// Resolve the authoritative or creation-time provider before applying component defaults.
	provider, providerSelected := defaultWorkloadProvider(ctx, dgd, req.Operation)

	// Default nil replicas on every operation so newly added components remain safe to expand.
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]

		// Default omitted replica counts before the controller expands component roles.
		if component.Replicas == nil {
			component.Replicas = ptr.To(int32(1))
		}

		// Default Grove's minimum available replicas only for Grove-selected DGDs.
		if providerSelected && provider == consts.WorkloadProviderGrove && component.MinAvailable == nil {
			component.MinAvailable = ptr.To(int32(1))
		}
	}

	// Stamp creation provenance independently from level-based provider defaulting.
	if req.Operation == admissionv1.Create {
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

func defaultWorkloadProvider(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	operation admissionv1.Operation,
) (string, bool) {
	// Keep existing selections authoritative and leave legacy updates for controller adoption.
	if operation != admissionv1.Create {
		if provider, exists := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]; exists {
			return provider, true
		}
		return "", false
	}

	// Derive every new DGD from user-facing routing intent, ignoring the controller-owned annotation.
	provider := consts.WorkloadProviderComponent

	// Select Grove when it is enabled and the DGD has not opted out.
	if features.MustGateFrom(ctx).Enabled(features.Grove) &&
		strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse {
		provider = consts.WorkloadProviderGrove
	}

	// Allocate annotation storage before materializing the selected provider.
	if dgd.Annotations == nil {
		dgd.Annotations = make(map[string]string)
	}
	dgd.Annotations[consts.KubeAnnotationWorkloadProvider] = provider
	return provider, true
}

// RegisterWithManager registers the defaulting webhook with the manager.
func (d *DGDDefaulter) RegisterWithManager(mgr manager.Manager, gate features.Gate) error {
	defaulter := internalwebhook.NewLeaseAwareDefaulter(d, internalwebhook.GetExcludedNamespaces())
	webhook := internalwebhook.WithGate(admission.
		WithCustomDefaulter(mgr.GetScheme(), &nvidiacomv1beta1.DynamoGraphDeployment{}, defaulter).
		WithRecoverPanic(true), gate)
	mgr.GetWebhookServer().Register(dgdDefaultingWebhookPath, webhook)
	return nil
}
