/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/

package api

import (
	"os"
	"path/filepath"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"sigs.k8s.io/yaml"
)

func TestMigratedCRDsUseBetaStorageAndServeAlpha(t *testing.T) {
	files := []string{
		"nvidia.com_dynamocomponentdeployments.yaml",
		"nvidia.com_dynamographdeployments.yaml",
		"nvidia.com_dynamographdeploymentrequests.yaml",
		"nvidia.com_dynamographdeploymentscalingadapters.yaml",
	}
	for _, file := range files {
		t.Run(file, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("..", "config", "crd", "bases", file))
			if err != nil {
				t.Fatal(err)
			}
			crd := &apiextensionsv1.CustomResourceDefinition{}
			if err := yaml.Unmarshal(data, crd); err != nil {
				t.Fatal(err)
			}
			versions := map[string]apiextensionsv1.CustomResourceDefinitionVersion{}
			for _, version := range crd.Spec.Versions {
				versions[version.Name] = version
			}
			alpha, alphaFound := versions["v1alpha1"]
			beta, betaFound := versions["v1beta1"]
			if !alphaFound || !betaFound {
				t.Fatalf("versions = %v, want v1alpha1 and v1beta1", versions)
			}
			if !alpha.Served || alpha.Storage {
				t.Fatalf("v1alpha1 served/storage = %t/%t, want true/false", alpha.Served, alpha.Storage)
			}
			if !beta.Served || !beta.Storage {
				t.Fatalf("v1beta1 served/storage = %t/%t, want true/true", beta.Served, beta.Storage)
			}
			if crd.Spec.Conversion == nil || crd.Spec.Conversion.Strategy != apiextensionsv1.WebhookConverter {
				t.Fatal("webhook conversion is not configured")
			}
		})
	}
}
