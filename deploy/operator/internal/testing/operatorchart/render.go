/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package operatorchart renders selected templates from the production operator Helm chart.
package operatorchart

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	goruntime "runtime"
	"strings"

	helmchart "helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/chartutil"
	"helm.sh/helm/v3/pkg/engine"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/yaml"
)

// Options describes the Helm release used to render an operator template.
type Options struct {
	ReleaseName string
	Namespace   string
	Values      map[string]any
}

// Render returns the Kubernetes objects produced by one operator chart template.
func Render(templateName string, options Options) ([]unstructured.Unstructured, error) {
	chart, err := loader.Load(operatorChartDirectory())
	if err != nil {
		return nil, fmt.Errorf("load operator Helm chart: %w", err)
	}
	if err := retainTemplate(chart, templateName); err != nil {
		return nil, err
	}
	values, err := chartutil.ToRenderValues(chart, chartutil.Values(options.Values), chartutil.ReleaseOptions{
		Name:      options.ReleaseName,
		Namespace: options.Namespace,
		IsInstall: true,
	}, nil)
	if err != nil {
		return nil, fmt.Errorf("prepare operator Helm values: %w", err)
	}
	rendered, err := engine.Engine{}.Render(chart, values)
	if err != nil {
		return nil, fmt.Errorf("render operator Helm chart: %w", err)
	}
	manifest, err := renderedTemplate(rendered, templateName)
	if err != nil {
		return nil, err
	}
	return decodeObjects(manifest)
}

func retainTemplate(chart *helmchart.Chart, templateName string) error {
	templates := chart.Templates[:0]
	found := false
	for _, template := range chart.Templates {
		if template.Name == templateName {
			found = true
			templates = append(templates, template)
			continue
		}
		if strings.HasPrefix(filepath.Base(template.Name), "_") {
			templates = append(templates, template)
		}
	}
	chart.Templates = templates
	if !found {
		return fmt.Errorf("operator Helm chart does not contain %s", templateName)
	}
	return nil
}

func renderedTemplate(rendered map[string]string, templateName string) (string, error) {
	for name, manifest := range rendered {
		if strings.HasSuffix(filepath.ToSlash(name), templateName) {
			return manifest, nil
		}
	}
	return "", fmt.Errorf("rendered operator Helm chart does not contain %s", templateName)
}

func decodeObjects(manifest string) ([]unstructured.Unstructured, error) {
	decoder := yaml.NewYAMLOrJSONDecoder(strings.NewReader(manifest), 4096)
	var objects []unstructured.Unstructured
	for {
		var object unstructured.Unstructured
		if err := decoder.Decode(&object); err != nil {
			if err == io.EOF {
				return objects, nil
			}
			return nil, fmt.Errorf("decode rendered operator manifest: %w", err)
		}
		if len(object.Object) != 0 {
			objects = append(objects, object)
		}
	}
}

func operatorChartDirectory() string {
	if directory := os.Getenv("OPERATOR_CHART_DIR"); directory != "" {
		return directory
	}
	_, file, _, _ := goruntime.Caller(0)
	operatorRoot := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
	return filepath.Join(operatorRoot, "..", "helm", "charts", "platform", "components", "operator")
}
