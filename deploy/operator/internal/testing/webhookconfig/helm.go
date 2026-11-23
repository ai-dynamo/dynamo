/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package webhookconfig provides admission registrations for test environments.
package webhookconfig

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorchart"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

// HelmConfigurations returns the admission registrations rendered from the
// production operator Helm chart.
func HelmConfigurations() ([]*admissionregistrationv1.MutatingWebhookConfiguration, []*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
	objects, err := operatorchart.Render("templates/webhook-configuration.yaml", operatorchart.Options{
		ReleaseName: "operatorenv",
		Namespace:   "operatorenv",
	})
	if err != nil {
		return nil, nil, err
	}
	return decodeWebhookConfigurations(objects)
}

func decodeWebhookConfigurations(objects []unstructured.Unstructured) ([]*admissionregistrationv1.MutatingWebhookConfiguration, []*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
	var mutating []*admissionregistrationv1.MutatingWebhookConfiguration
	var validating []*admissionregistrationv1.ValidatingWebhookConfiguration
	for _, object := range objects {
		switch object.GetKind() {
		case "MutatingWebhookConfiguration":
			webhook := &admissionregistrationv1.MutatingWebhookConfiguration{}
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(object.Object, webhook); err != nil {
				return nil, nil, fmt.Errorf("decode mutating webhook configuration %q: %w", object.GetName(), err)
			}
			mutating = append(mutating, webhook)
		case "ValidatingWebhookConfiguration":
			webhook := &admissionregistrationv1.ValidatingWebhookConfiguration{}
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(object.Object, webhook); err != nil {
				return nil, nil, fmt.Errorf("decode validating webhook configuration %q: %w", object.GetName(), err)
			}
			validating = append(validating, webhook)
		}
	}
	if len(mutating) == 0 || len(validating) == 0 {
		return nil, nil, fmt.Errorf("rendered webhook configuration contains %d mutating and %d validating webhooks", len(mutating), len(validating))
	}
	return mutating, validating, nil
}
