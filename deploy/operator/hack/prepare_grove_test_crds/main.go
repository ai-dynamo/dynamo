// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Emit pinned Grove schemas for tests that do not run the Grove controller.
package main

import (
	"fmt"
	"os"

	grovecrds "github.com/ai-dynamo/grove/operator/api/core/crds"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"sigs.k8s.io/yaml"
)

func main() {
	// Fail before emitting anything if the pinned API no longer serves Dynamo's version.
	crds, err := testCRDs()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// Stream manifests directly into the explicitly selected cluster's installer.
	for _, crd := range crds {
		data, err := yaml.Marshal(crd)
		if err == nil {
			_, err = fmt.Fprintf(os.Stdout, "---\n%s", data)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
}

func testCRDs() ([]*apiextensionsv1.CustomResourceDefinition, error) {
	// Preserve each pinned alpha schema while removing the dependency on Grove's converter.
	result := make([]*apiextensionsv1.CustomResourceDefinition, 0, 5)
	for _, manifest := range []string{
		grovecrds.PodCliqueSetCRD(),
		grovecrds.PodCliqueCRD(),
		grovecrds.PodCliqueScalingGroupCRD(),
		grovecrds.ClusterTopologyCRD(),
		grovecrds.PodGangMapCRD(),
	} {
		crd := &apiextensionsv1.CustomResourceDefinition{}
		if err := yaml.Unmarshal([]byte(manifest), crd); err != nil {
			return nil, err
		}
		versions := make([]apiextensionsv1.CustomResourceDefinitionVersion, 0, 1)
		for _, version := range crd.Spec.Versions {
			if version.Name == "v1alpha1" && version.Served {
				version.Storage = true
				versions = append(versions, version)
			}
		}
		if len(versions) != 1 {
			return nil, fmt.Errorf("%s: expected one served v1alpha1 schema", crd.Name)
		}
		crd.Spec.Versions = versions
		crd.Spec.Conversion = &apiextensionsv1.CustomResourceConversion{Strategy: apiextensionsv1.NoneConverter}
		result = append(result, crd)
	}
	return result, nil
}
