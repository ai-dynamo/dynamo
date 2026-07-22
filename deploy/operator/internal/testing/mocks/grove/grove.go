/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package grove provides transparent PodCliqueSet admission handlers until
// Grove exposes its production handlers as an importable package.
package grove

import (
	"context"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/webhookconfig"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	groveGroup          = "grove.io"
	groveResource       = "podcliquesets"
	groveWebhookName    = "webhook-service"
	groveWebhookNS      = "system"
	groveMutatingPath   = "/webhooks/default-podcliqueset"
	groveValidatingPath = "/webhooks/validate-podcliqueset"
)

type handler struct{}

// Setup registers transparent handlers at the production Grove webhook paths.
func Setup(mgr ctrl.Manager) error {
	h := &handler{}
	mgr.GetWebhookServer().Register(
		groveMutatingPath,
		admission.WithCustomDefaulter(mgr.GetScheme(), &grovev1alpha1.PodCliqueSet{}, h).WithRecoverPanic(true),
	)
	mgr.GetWebhookServer().Register(
		groveValidatingPath,
		admission.WithCustomValidator(mgr.GetScheme(), &grovev1alpha1.PodCliqueSet{}, h).WithRecoverPanic(true),
	)
	return nil
}

func (*handler) Default(context.Context, runtime.Object) error {
	return nil
}

func (*handler) ValidateCreate(context.Context, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func (*handler) ValidateUpdate(context.Context, runtime.Object, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func (*handler) ValidateDelete(context.Context, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

// Configurations returns focused registrations matching Grove's PodCliqueSet
// admission configurations.
func Configurations() webhookconfig.Configurations {
	fail := admissionregistrationv1.Fail
	exact := admissionregistrationv1.Exact
	none := admissionregistrationv1.SideEffectClassNone
	namespaced := admissionregistrationv1.NamespacedScope
	operations := []admissionregistrationv1.OperationType{
		admissionregistrationv1.Create,
		admissionregistrationv1.Update,
	}
	rule := admissionregistrationv1.RuleWithOperations{
		Operations: operations,
		Rule: admissionregistrationv1.Rule{
			APIGroups:   []string{groveGroup},
			APIVersions: []string{"v1alpha1"},
			Resources:   []string{groveResource},
			Scope:       &namespaced,
		},
	}
	return webhookconfig.Configurations{
		Mutating: []*admissionregistrationv1.MutatingWebhookConfiguration{{
			ObjectMeta: metav1.ObjectMeta{Name: "podcliqueset-defaulting-webhook"},
			Webhooks: []admissionregistrationv1.MutatingWebhook{{
				Name:                    "pcs.defaulting.webhooks.grove.io",
				AdmissionReviewVersions: []string{"v1"},
				FailurePolicy:           &fail,
				MatchPolicy:             &exact,
				SideEffects:             &none,
				Rules:                   []admissionregistrationv1.RuleWithOperations{rule},
				ClientConfig:            serviceClientConfig(groveMutatingPath),
			}},
		}},
		Validating: []*admissionregistrationv1.ValidatingWebhookConfiguration{{
			ObjectMeta: metav1.ObjectMeta{Name: "podcliqueset-validating-webhook"},
			Webhooks: []admissionregistrationv1.ValidatingWebhook{{
				Name:                    "pcs.validating.webhooks.grove.io",
				AdmissionReviewVersions: []string{"v1"},
				FailurePolicy:           &fail,
				MatchPolicy:             &exact,
				SideEffects:             &none,
				Rules:                   []admissionregistrationv1.RuleWithOperations{rule},
				ClientConfig:            serviceClientConfig(groveValidatingPath),
			}},
		}},
	}
}

func serviceClientConfig(path string) admissionregistrationv1.WebhookClientConfig {
	return admissionregistrationv1.WebhookClientConfig{Service: &admissionregistrationv1.ServiceReference{
		Namespace: groveWebhookNS,
		Name:      groveWebhookName,
		Path:      &path,
	}}
}
