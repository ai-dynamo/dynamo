/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dgdoverride

import (
	"fmt"
	"sync"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	operatorcrd "github.com/ai-dynamo/dynamo/deploy/operator/config/crd"
	apixv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/controller/openapi/builder"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"sigs.k8s.io/yaml"
)

type dgdSchemas struct {
	typeConverter    managedfields.TypeConverter
	rootByAPIVersion map[string]*apixv1.JSONSchemaProps
}

var loadDGDSchemas = sync.OnceValues(buildDGDSchemas)

func buildDGDSchemas() (*dgdSchemas, error) {
	definition := &apixv1.CustomResourceDefinition{}
	if err := yaml.Unmarshal([]byte(operatorcrd.DynamoGraphDeploymentDefinition()), definition); err != nil {
		return nil, fmt.Errorf("decode embedded DGD CRD: %w", err)
	}

	rootByAPIVersion := make(map[string]*apixv1.JSONSchemaProps, 2)
	for _, apiVersion := range []string{
		nvidiacomv1alpha1.GroupVersion.Version,
		nvidiacomv1beta1.GroupVersion.Version,
	} {
		for i := range definition.Spec.Versions {
			version := &definition.Spec.Versions[i]
			if version.Name == apiVersion && version.Schema != nil && version.Schema.OpenAPIV3Schema != nil {
				rootByAPIVersion[apiVersion] = version.Schema.OpenAPIV3Schema
				break
			}
		}
		if rootByAPIVersion[apiVersion] == nil {
			return nil, fmt.Errorf("embedded DGD CRD has no schema for %s", apiVersion)
		}
	}

	alphaDocument, err := builder.BuildOpenAPIV3(
		definition,
		nvidiacomv1alpha1.GroupVersion.Version,
		builder.Options{},
	)
	if err != nil {
		return nil, fmt.Errorf("build DGD %s OpenAPI schema: %w", nvidiacomv1alpha1.GroupVersion.Version, err)
	}
	betaDocument, err := builder.BuildOpenAPIV3(
		definition,
		nvidiacomv1beta1.GroupVersion.Version,
		builder.Options{},
	)
	if err != nil {
		return nil, fmt.Errorf("build DGD %s OpenAPI schema: %w", nvidiacomv1beta1.GroupVersion.Version, err)
	}

	merged, err := builder.MergeSpecsV3(alphaDocument, betaDocument)
	if err != nil {
		return nil, fmt.Errorf("merge DGD OpenAPI schemas: %w", err)
	}
	if merged.Components == nil || len(merged.Components.Schemas) == 0 {
		return nil, fmt.Errorf("embedded DGD CRD produced no OpenAPI schemas")
	}

	converter, err := managedfields.NewTypeConverter(
		merged.Components.Schemas,
		definition.Spec.PreserveUnknownFields,
	)
	if err != nil {
		return nil, fmt.Errorf("create DGD structural type converter: %w", err)
	}
	return &dgdSchemas{
		typeConverter:    converter,
		rootByAPIVersion: rootByAPIVersion,
	}, nil
}
