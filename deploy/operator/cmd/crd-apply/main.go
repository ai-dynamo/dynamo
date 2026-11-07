/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/yaml"
)

const (
	fieldManager       = "dynamo-crd-apply"
	versionAnnotation  = "dynamo.nvidia.com/operator-version"
	dgdrV1Beta2Flag    = "dgdr-v1beta2-enabled"
	apiV1Alpha1Version = "v1alpha1"
	dgdrV1Beta2Version = "v1beta2"

	dgdrCRDName      = "dynamographdeploymentrequests.nvidia.com"
	dgdrRunCRDName   = "dynamographdeploymentruns.nvidia.com"
	dgdrCandidateCRD = "dynamographdeploymentcandidates.nvidia.com"
)

func main() {
	crdsDir := flag.String("crds-dir", "/opt/dynamo-operator/crds/", "Directory containing CRD YAML files")
	version := flag.String("version", "", "Operator version to stamp on CRDs")
	lpxEnabled := flag.Bool("lpx-enabled", false, "Install the experimental LPXGraphDeployment CRD")
	dgdrV1Beta2Enabled := flag.Bool(dgdrV1Beta2Flag, false, "Install the experimental DGDR v1beta2 API schemas")
	conversionWebhookServiceName := flag.String(
		"conversion-webhook-service-name",
		"",
		"Service name for CRD conversion webhooks",
	)
	conversionWebhookServiceNamespace := flag.String(
		"conversion-webhook-service-namespace",
		"",
		"Service namespace for CRD conversion webhooks",
	)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))
	log := ctrl.Log.WithName("crd-apply")
	if *dgdrV1Beta2Enabled {
		log.Info(
			"WARNING: enabled experimental DGDR v1beta2 API schemas; use only on development clusters. "+
				"DynamoGraphDeploymentRun and DynamoGraphDeploymentCandidate objects are persisted in etcd, "+
				"and incompatible schema changes may block future CRD upgrades",
			"flag", dgdrV1Beta2Flag,
		)
	}

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
		if crd.Name == "lpxgraphdeployments.nvidia.com" && !*lpxEnabled {
			continue
		}
		if !*dgdrV1Beta2Enabled && !configureDGDRV1Beta2CRD(crd) {
			log.Info("Skipped disabled experimental CRD", "crd", crd.Name, "flag", dgdrV1Beta2Flag)
			continue
		}

		if *version != "" {
			if crd.Annotations == nil {
				crd.Annotations = make(map[string]string)
			}
			crd.Annotations[versionAnnotation] = *version
		}
		if err := configureConversionWebhookService(
			crd,
			*conversionWebhookServiceName,
			*conversionWebhookServiceNamespace,
		); err != nil {
			log.Error(err, "unable to configure CRD conversion webhook service", "crd", crd.Name)
			os.Exit(1)
		}

		// Compact JSON avoids expanding large CRD schemas into oversized YAML requests.
		patchData, err := json.Marshal(crd)
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
				Force:        new(true),
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

// configureDGDRV1Beta2CRD projects generated CRDs to the disabled API shape.
func configureDGDRV1Beta2CRD(crd *apiextensionsv1.CustomResourceDefinition) bool {
	switch crd.Name {
	case dgdrRunCRDName, dgdrCandidateCRD:
		return false
	case dgdrCRDName:
		crd.Spec.Versions = slices.DeleteFunc(
			crd.Spec.Versions,
			func(version apiextensionsv1.CustomResourceDefinitionVersion) bool {
				return version.Name == dgdrV1Beta2Version
			},
		)
		for i := range crd.Spec.Versions {
			removeDGDRV1Beta2Envelope(&crd.Spec.Versions[i])
		}
	}
	return true
}

func removeDGDRV1Beta2Envelope(version *apiextensionsv1.CustomResourceDefinitionVersion) {
	if version.Schema == nil || version.Schema.OpenAPIV3Schema == nil {
		return
	}

	root := version.Schema.OpenAPIV3Schema
	root.XValidations = slices.DeleteFunc(
		root.XValidations,
		func(rule apiextensionsv1.ValidationRule) bool {
			return strings.Contains(rule.Rule, dgdrV1Beta2Version)
		},
	)
	spec := root.Properties["spec"]
	delete(spec.Properties, dgdrV1Beta2Version)
	spec.XValidations = slices.DeleteFunc(
		spec.XValidations,
		func(rule apiextensionsv1.ValidationRule) bool {
			return strings.Contains(rule.Rule, dgdrV1Beta2Version)
		},
	)
	switch version.Name {
	case apiV1Alpha1Version:
		spec.Required = appendRequired(spec.Required, "model", "backend", "profilingConfig")
	case "v1beta1":
		spec.Required = appendRequired(spec.Required, "model")
	}
	root.Properties["spec"] = spec

	status := root.Properties["status"]
	delete(status.Properties, dgdrV1Beta2Version)
	status.XValidations = slices.DeleteFunc(
		status.XValidations,
		func(rule apiextensionsv1.ValidationRule) bool {
			return strings.Contains(rule.Rule, dgdrV1Beta2Version)
		},
	)
	root.Properties["status"] = status
}

func appendRequired(required []string, fields ...string) []string {
	for _, field := range fields {
		if !slices.Contains(required, field) {
			required = append(required, field)
		}
	}
	return required
}

func configureConversionWebhookService(
	crd *apiextensionsv1.CustomResourceDefinition,
	serviceName string,
	serviceNamespace string,
) error {
	if crd.Spec.Conversion == nil ||
		crd.Spec.Conversion.Webhook == nil ||
		crd.Spec.Conversion.Webhook.ClientConfig == nil ||
		crd.Spec.Conversion.Webhook.ClientConfig.Service == nil {
		return nil
	}
	if serviceName == "" {
		return fmt.Errorf("conversion webhook service name is required")
	}
	if serviceNamespace == "" {
		return fmt.Errorf("conversion webhook service namespace is required")
	}

	// Override service coordinates because these might be customized through Helm values.
	crd.Spec.Conversion.Webhook.ClientConfig.Service.Name = serviceName
	crd.Spec.Conversion.Webhook.ClientConfig.Service.Namespace = serviceNamespace

	// The main operator process patches caBundle after certificate setup.
	crd.Spec.Conversion.Webhook.ClientConfig.CABundle = nil
	return nil
}
