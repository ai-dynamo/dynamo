/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/webhookconfig"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	lwswebhooks "sigs.k8s.io/lws/pkg/webhooks"
	dswebhooks "sigs.k8s.io/lws/pkg/webhooks/disaggregatedset"
)

const (
	lwsGroup          = "leaderworkerset.x-k8s.io"
	lwsResource       = "leaderworkersets"
	lwsWebhookName    = "webhook-service"
	lwsWebhookNS      = "system"
	lwsMutatingPath   = "/mutate-leaderworkerset-x-k8s-io-v1-leaderworkerset"
	lwsValidatingPath = "/validate-leaderworkerset-x-k8s-io-v1-leaderworkerset"
)

// Setup registers the production LeaderWorkerSet and DisaggregatedSet admission handlers.
func Setup(mgr ctrl.Manager) error {
	if err := lwswebhooks.SetupLeaderWorkerSetWebhook(mgr); err != nil {
		return err
	}
	return dswebhooks.SetupDisaggregatedSetWebhook(mgr)
}

// Configurations returns the workload admission registrations from upstream markers.
// Pod webhooks stay excluded because these tests stop at immediate workloads.
func Configurations() webhookconfig.Configurations {
	fail := admissionregistrationv1.Fail
	none := admissionregistrationv1.SideEffectClassNone
	operations := []admissionregistrationv1.OperationType{
		admissionregistrationv1.Create,
		admissionregistrationv1.Update,
	}
	rule := admissionregistrationv1.RuleWithOperations{
		Operations:  operations,
		APIGroups:   []string{lwsGroup},
		APIVersions: []string{"v1"},
		Resources:   []string{lwsResource},
	}
	return webhookconfig.Configurations{
		Mutating: []*admissionregistrationv1.MutatingWebhookConfiguration{{
			Name: "lws-mutating-webhook-configuration",
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
			Name: "lws-validating-webhook-configuration",
			Webhooks: []admissionregistrationv1.ValidatingWebhook{{
				Name:                    "vleaderworkerset.kb.io",
				AdmissionReviewVersions: []string{"v1"},
				FailurePolicy:           &fail,
				SideEffects:             &none,
				Rules:                   []admissionregistrationv1.RuleWithOperations{rule},
				ClientConfig:            serviceClientConfig(lwsValidatingPath),
			}, {
				Name:                    "vdisaggregatedset.kb.io",
				AdmissionReviewVersions: []string{"v1"},
				FailurePolicy:           &fail,
				SideEffects:             &none,
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations:  operations,
					APIGroups:   []string{"disaggregatedset.x-k8s.io"},
					APIVersions: []string{"v1"},
					Resources:   []string{"disaggregatedsets"},
				}},
				ClientConfig: serviceClientConfig("/validate-disaggregatedset-x-k8s-io-v1-disaggregatedset"),
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
