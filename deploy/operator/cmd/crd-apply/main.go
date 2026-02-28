/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/yaml"
)

const (
	fieldManager      = "dynamo-crd-apply"
	versionAnnotation = "dynamo.nvidia.com/operator-version"
)

func main() {
	crdsDir := flag.String("crds-dir", "/opt/dynamo-operator/crds/", "Directory containing CRD YAML files")
	version := flag.String("version", "", "Operator version to stamp on CRDs")
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))
	log := ctrl.Log.WithName("crd-apply")

	config, err := ctrl.GetConfig()
	if err != nil {
		log.Error(err, "unable to get kubernetes config")
		os.Exit(1)
	}

	client, err := apiextensionsclient.NewForConfig(config)
	if err != nil {
		log.Error(err, "unable to create apiextensions client")
		os.Exit(1)
	}

	entries, err := os.ReadDir(*crdsDir)
	if err != nil {
		log.Error(err, "unable to read CRDs directory", "dir", *crdsDir)
		os.Exit(1)
	}

	ctx := context.Background()
	var applied int

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}

		filePath := filepath.Join(*crdsDir, entry.Name())
		data, err := os.ReadFile(filePath)
		if err != nil {
			log.Error(err, "unable to read CRD file", "file", filePath)
			os.Exit(1)
		}

		crd := &apiextensionsv1.CustomResourceDefinition{}
		if err := yaml.Unmarshal(data, crd); err != nil {
			log.Error(err, "unable to unmarshal CRD", "file", filePath)
			os.Exit(1)
		}

		if *version != "" {
			if crd.Annotations == nil {
				crd.Annotations = make(map[string]string)
			}
			crd.Annotations[versionAnnotation] = *version
		}

		patchData, err := yaml.Marshal(crd)
		if err != nil {
			log.Error(err, "unable to marshal CRD for patch", "crd", crd.Name)
			os.Exit(1)
		}

		_, err = client.ApiextensionsV1().CustomResourceDefinitions().Patch(
			ctx,
			crd.Name,
			types.ApplyPatchType,
			patchData,
			metav1.PatchOptions{
				FieldManager: fieldManager,
				Force:        ptr.To(true),
			},
		)
		if err != nil {
			log.Error(err, "unable to apply CRD", "crd", crd.Name)
			os.Exit(1)
		}

		log.Info("Applied CRD", "crd", crd.Name)
		applied++
	}

	if applied == 0 {
		fmt.Fprintf(os.Stderr, "WARNING: no CRD files found in %s\n", *crdsDir)
		os.Exit(1)
	}

	log.Info("CRD apply complete", "applied", applied)
}
