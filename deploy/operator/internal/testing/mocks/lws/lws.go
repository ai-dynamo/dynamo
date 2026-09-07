/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/webhookconfig"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	lwswebhooks "sigs.k8s.io/lws/pkg/webhooks"
)

const (
	lwsGroup          = "leaderworkerset.x-k8s.io"
	lwsResource       = "leaderworkersets"
	lwsWebhookName    = "webhook-service"
	lwsWebhookNS      = "system"
	lwsMutatingPath   = "/mutate-leaderworkerset-x-k8s-io-v1-leaderworkerset"
	lwsValidatingPath = "/validate-leaderworkerset-x-k8s-io-v1-leaderworkerset"
)

// Setup registers the production LeaderWorkerSet admission handlers.
func Setup(mgr ctrl.Manager) error {
	return lwswebhooks.SetupLeaderWorkerSetWebhook(mgr)
}

// Configurations returns the admission registrations generated from the
// LeaderWorkerSet webhook markers, restricted to LeaderWorkerSet.
func Configurations() webhookconfig.Configurations {
	fail := admissionregistrationv1.Fail
	none := admissionregistrationv1.SideEffectClassNone
	operations := []admissionregistrationv1.OperationType{
		admissionregistrationv1.Create,
		admissionregistrationv1.Update,
	}
	rule := admissionregistrationv1.RuleWithOperations{
		Operations: operations,
		Rule: admissionregistrationv1.Rule{
			APIGroups:   []string{lwsGroup},
			APIVersions: []string{"v1"},
			Resources:   []string{lwsResource},
		},
	}
	return webhookconfig.Configurations{
		Mutating: []*admissionregistrationv1.MutatingWebhookConfiguration{{
			ObjectMeta: metav1.ObjectMeta{Name: "lws-mutating-webhook-configuration"},
			Webhooks: []admissionregistrationv1.MutatingWebhook{{
				Name:                    "mleaderworkerset.kb.io",
				AdmissionReviewVersions: []string{"v1"},
				FailurePolicy:           &fail,
				SideEffects:             &none,
				Rules:                   []admissionregistrationv1.RuleWithOperations{rule},
				ClientConfig:            serviceClientConfig(lwsMutatingPath),
			}},
		}},
		Validating: []*admissionregistrationv1.ValidatingWebhookConfiguration{{
			ObjectMeta: metav1.ObjectMeta{Name: "lws-validating-webhook-configuration"},
			Webhooks: []admissionregistrationv1.ValidatingWebhook{{
				Name:                    "vleaderworkerset.kb.io",
				AdmissionReviewVersions: []string{"v1"},
				FailurePolicy:           &fail,
				SideEffects:             &none,
				Rules:                   []admissionregistrationv1.RuleWithOperations{rule},
				ClientConfig:            serviceClientConfig(lwsValidatingPath),
			}},
		}},
	}
}

func serviceClientConfig(path string) admissionregistrationv1.WebhookClientConfig {
	return admissionregistrationv1.WebhookClientConfig{Service: &admissionregistrationv1.ServiceReference{
		Namespace: lwsWebhookNS,
		Name:      lwsWebhookName,
		Path:      &path,
	}}
}
