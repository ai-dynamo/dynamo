/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package operatorenv

import (
	"fmt"
	"io"
	"path/filepath"
	"strings"

	helmchart "helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/chartutil"
	"helm.sh/helm/v3/pkg/engine"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/yaml"
)

const webhookConfigurationTemplate = "templates/webhook-configuration.yaml"

func helmWebhookConfigurations() ([]*admissionregistrationv1.MutatingWebhookConfiguration, []*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
	chart, err := loader.Load(operatorChartDirectory())
	if err != nil {
		return nil, nil, fmt.Errorf("load operator Helm chart: %w", err)
	}
	if err := retainWebhookTemplate(chart); err != nil {
		return nil, nil, err
	}
	values, err := chartutil.ToRenderValues(chart, chartutil.Values{}, chartutil.ReleaseOptions{
		Name:      "operatorenv",
		Namespace: "operatorenv",
		IsInstall: true,
	}, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("prepare operator Helm values: %w", err)
	}
	rendered, err := engine.Engine{}.Render(chart, values)
	if err != nil {
		return nil, nil, fmt.Errorf("render operator Helm chart: %w", err)
	}
	manifest, err := webhookConfiguration(rendered)
	if err != nil {
		return nil, nil, err
	}
	return decodeWebhookConfigurations(manifest)
}

func retainWebhookTemplate(chart *helmchart.Chart) error {
	templates := chart.Templates[:0]
	foundWebhookTemplate := false
	for _, template := range chart.Templates {
		if template.Name == webhookConfigurationTemplate {
			foundWebhookTemplate = true
			templates = append(templates, template)
			continue
		}
		if strings.HasPrefix(filepath.Base(template.Name), "_") {
			templates = append(templates, template)
		}
	}
	chart.Templates = templates
	if !foundWebhookTemplate {
		return fmt.Errorf("operator Helm chart does not contain %s", webhookConfigurationTemplate)
	}
	return nil
}

func operatorChartDirectory() string {
	return filepath.Join(operatorRoot(), "..", "helm", "charts", "platform", "components", "operator")
}

func webhookConfiguration(rendered map[string]string) (string, error) {
	for name, manifest := range rendered {
		if strings.HasSuffix(filepath.ToSlash(name), webhookConfigurationTemplate) {
			return manifest, nil
		}
	}
	return "", fmt.Errorf("rendered operator Helm chart does not contain %s", webhookConfigurationTemplate)
}

func decodeWebhookConfigurations(manifest string) ([]*admissionregistrationv1.MutatingWebhookConfiguration, []*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
	decoder := yaml.NewYAMLOrJSONDecoder(strings.NewReader(manifest), 4096)
	var mutating []*admissionregistrationv1.MutatingWebhookConfiguration
	var validating []*admissionregistrationv1.ValidatingWebhookConfiguration
	for {
		var object unstructured.Unstructured
		if err := decoder.Decode(&object); err != nil {
			if err == io.EOF {
				break
			}
			return nil, nil, fmt.Errorf("decode rendered webhook configuration: %w", err)
		}
		if len(object.Object) == 0 {
			continue
		}
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
