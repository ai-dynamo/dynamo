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
	"slices"
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
	dgdrV1Beta2Gate   = "DGDRV1Beta2"

	dgdrCRDName      = "dynamographdeploymentrequests.nvidia.com"
	dgdrRunCRDName   = "dynamographdeploymentruns.nvidia.com"
	dgdrCandidateCRD = "dynamographdeploymentcandidates.nvidia.com"
)

func main() {
	crdsDir := flag.String("crds-dir", "/opt/dynamo-operator/crds/", "Directory containing CRD YAML files")
	version := flag.String("version", "", "Operator version to stamp on CRDs")
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
	featureGates := flag.String(
		"feature-gates",
		"",
		"Comma-separated feature gates in Name=true|false form",
	)
	flag.Parse()

	gates, err := parseFeatureGates(*featureGates)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid feature gates: %v\n", err)
		os.Exit(1)
	}

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
		if !gates[dgdrV1Beta2Gate] && !configureDGDRV1Beta2CRD(crd) {
			log.Info("Skipped CRD disabled by feature gate", "crd", crd.Name, "featureGate", dgdrV1Beta2Gate)
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

func parseFeatureGates(value string) (map[string]bool, error) {
	gates := map[string]bool{dgdrV1Beta2Gate: false}
	if value == "" {
		return gates, nil
	}
	for _, item := range strings.Split(value, ",") {
		name, rawValue, found := strings.Cut(item, "=")
		if !found || name == "" {
			return nil, fmt.Errorf("%q must use Name=true|false format", item)
		}
		if _, known := gates[name]; !known {
			return nil, fmt.Errorf("unknown feature gate %q", name)
		}
		switch rawValue {
		case "true":
			gates[name] = true
		case "false":
			gates[name] = false
		default:
			return nil, fmt.Errorf("feature gate %q must be true or false", name)
		}
	}
	return gates, nil
}

// configureDGDRV1Beta2CRD removes experimental API discovery when the gate is disabled.
// Existing standalone CRDs are skipped, not deleted, to avoid destructive upgrades.
func configureDGDRV1Beta2CRD(crd *apiextensionsv1.CustomResourceDefinition) bool {
	switch crd.Name {
	case dgdrRunCRDName, dgdrCandidateCRD:
		return false
	case dgdrCRDName:
		crd.Spec.Versions = slices.DeleteFunc(crd.Spec.Versions, func(version apiextensionsv1.CustomResourceDefinitionVersion) bool {
			return version.Name == "v1beta2"
		})
	}
	return true
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
