// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation

import (
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	apitest "k8s.io/apiextensions-apiserver/pkg/test"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestComponentRoleStatusSparsePodSpecSchema(t *testing.T) {
	// Spec admission removes status, so exercise its generated pruning/defaulting contract directly.
	crd := apitest.MustLoadManifest[apiextensionsv1.CustomResourceDefinition](t, "../../../config/crd/bases/nvidia.com_dynamocomponentdeployments.yaml")
	for _, version := range crd.Spec.Versions {
		t.Run(version.Name, func(t *testing.T) {
			t.Log("Load the real native status schema without inventing Pod defaults or validation")
			var schema apiextensions.JSONSchemaProps
			require.NoError(t, apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(version.Schema.OpenAPIV3Schema, &schema, nil))
			structural, err := structuralschema.NewStructural(&schema)
			require.NoError(t, err)
			field := "component"
			if version.Name == "v1alpha1" {
				field = "service"
			}
			object := map[string]any{"status": map[string]any{field: map[string]any{"roles": []any{
				map[string]any{"name": "main", "podSpec": map[string]any{
					"initContainers": []any{map[string]any{"name": "runtime", "env": []any{
						map[string]any{"name": "EMPTY"},
						map[string]any{"name": "RANK", "value": "$(PROVIDER_RANK)"},
						map[string]any{"name": "POD", "valueFrom": map[string]any{"fieldRef": map[string]any{"fieldPath": "metadata.name"}}},
					}}},
				}},
			}}}}
			original := runtime.DeepCopyJSON(object)

			t.Log("Preserve a sparse native PodSpec and EnvVar list without adding or pruning fields")
			defaulting.Default(object, structural)
			pruning.Prune(object, structural, true)
			require.Equal(t, original, object)
		})
	}
}
