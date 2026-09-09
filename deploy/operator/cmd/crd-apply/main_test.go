/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
)

func TestCRDApplyInstallsGeneratedSchemas(t *testing.T) {
	const objectType = "object"

	t.Log("Start a local API server with its default request and storage limits")
	env := &envtest.Environment{UseExistingCluster: ptr.To(false)}
	config, err := env.Start()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := env.Stop(); err != nil {
			t.Error(err)
		}
	})

	t.Log("Give the installer credentials for only this ephemeral API server")
	dir := t.TempDir()
	kubeconfig := filepath.Join(dir, "kubeconfig")
	if err := clientcmd.WriteToFile(clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{"test": {
			Server: config.Host, CertificateAuthorityData: config.CAData,
		}},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{"test": {
			ClientCertificateData: config.CertData, ClientKeyData: config.KeyData,
		}},
		Contexts:       map[string]*clientcmdapi.Context{"test": {Cluster: "test", AuthInfo: "test"}},
		CurrentContext: "test",
	}, kubeconfig); err != nil {
		t.Fatal(err)
	}

	t.Log("Build the production CRD installer")
	installer := filepath.Join(dir, "crd-apply")
	if output, err := exec.CommandContext(t.Context(), "go", "build", "-o", installer, ".").CombinedOutput(); err != nil {
		t.Fatalf("build installer: %v\n%s", err, output)
	}

	t.Log("Install and reapply every generated CRD, optionally upgrading an external baseline")
	crdDirectories := []string{"../../config/crd/bases", "../../config/crd/bases"}
	if baseline := os.Getenv("CRD_APPLY_BASELINE_DIR"); baseline != "" {
		crdDirectories = append([]string{baseline}, crdDirectories...)
	}
	for _, crdDirectory := range crdDirectories {
		command := exec.CommandContext(t.Context(), installer,
			"--crds-dir", crdDirectory,
			"--version", "schema-test",
			"--conversion-webhook-service-name", "dynamo-operator-webhook-service",
			"--conversion-webhook-service-namespace", "dynamo-system",
		)
		command.Env = append(os.Environ(), "KUBECONFIG="+kubeconfig)
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("apply CRDs: %v\n%s", err, output)
		}
	}

	t.Log("Verify installed schemas preserve DGD placement, keep LPX status grouped, and keep private child status flat")
	client, err := apiextensionsclient.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"dynamographdeployments.nvidia.com", "dynamocomponentdeployments.nvidia.com", "lpxgraphdeployments.nvidia.com",
	} {
		crd, err := client.ApiextensionsV1().CustomResourceDefinitions().Get(t.Context(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if crd.Annotations[versionAnnotation] != "schema-test" {
			t.Errorf("CRD %s is missing the installer version", name)
		}
		if name == "lpxgraphdeployments.nvidia.com" {
			t.Log("Keep the private LGD exclusively alpha, without conversion")
			require.Len(t, crd.Spec.Versions, 1)
			version := crd.Spec.Versions[0]
			require.Equal(t, "v1alpha1", version.Name)
			require.True(t, version.Served)
			require.True(t, version.Storage)
			require.Equal(t, []string{version.Name}, crd.Status.StoredVersions)
			require.Equal(t,
				&apiextensionsv1.CustomResourceConversion{Strategy: apiextensionsv1.NoneConverter},
				crd.Spec.Conversion)
		}
		if name == "dynamocomponentdeployments.nvidia.com" {
			t.Log("Retain ordinary template schemas and documentation in both served DCD versions")
			for _, version := range crd.Spec.Versions {
				templateField := "podTemplate"
				if version.Name == "v1alpha1" {
					templateField = "extraPodSpec"
				}
				template := version.Schema.OpenAPIV3Schema.Properties["spec"].Properties[templateField]
				if template.Type != objectType || template.Description == "" {
					t.Errorf("%s %s is missing the documented %s schema", name, version.Name, templateField)
				}
			}
			continue
		}
		for _, version := range crd.Spec.Versions {
			status := version.Schema.OpenAPIV3Schema.Properties["status"]
			if name == "dynamographdeployments.nvidia.com" {
				if _, exists := status.Properties["modelDownload"]; exists {
					t.Errorf("%s %s retains status.modelDownload", name, version.Name)
				}
				if status.Properties["placement"].Type != objectType {
					t.Errorf("%s %s is missing the established status.placement projection", name, version.Name)
				}
				status = status.Properties["lpx"]
			} else if _, exists := status.Properties["lpx"]; exists {
				t.Errorf("%s %s unexpectedly groups private child status", name, version.Name)
			}
			for _, field := range []string{"modelDownload", "placement"} {
				if status.Properties[field].Type != objectType {
					t.Errorf("%s %s is missing the typed %s projection", name, version.Name, field)
				}
			}
		}
	}
}
