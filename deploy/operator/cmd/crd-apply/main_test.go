/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	structuraldefaulting "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/listtype"
	apiextensionsvalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apitest "k8s.io/apiextensions-apiserver/pkg/test"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/yaml"
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

	t.Log("Install with LPX disabled, enable it, then disable it without changing its CRD")
	client, err := apiextensionsclient.NewForConfig(config)
	require.NoError(t, err)
	var lpxCRD *apiextensionsv1.CustomResourceDefinition
	crdDirectories := []string{"../../config/crd/bases", "../../config/crd/bases", "../../config/crd/bases"}
	if baseline := os.Getenv("CRD_APPLY_BASELINE_DIR"); baseline != "" {
		crdDirectories = append([]string{baseline}, crdDirectories...)
	}
	for index, crdDirectory := range crdDirectories {
		lpxEnabled := index == len(crdDirectories)-2
		command := exec.CommandContext(t.Context(), installer,
			"--crds-dir", crdDirectory,
			"--version", "schema-test",
			"--lpx-enabled="+strconv.FormatBool(lpxEnabled),
			"--conversion-webhook-service-name", "dynamo-operator-webhook-service",
			"--conversion-webhook-service-namespace", "dynamo-system",
		)
		command.Env = append(os.Environ(), "KUBECONFIG="+kubeconfig)
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("apply CRDs: %v\n%s", err, output)
		}

		current, err := client.ApiextensionsV1().CustomResourceDefinitions().Get(
			t.Context(), "lpxgraphdeployments.nvidia.com", metav1.GetOptions{})
		switch {
		case lpxEnabled:
			require.NoError(t, err)
			lpxCRD = current
		case lpxCRD == nil:
			require.True(t, apierrors.IsNotFound(err), "LPX CRD must not be installed while disabled: %v", err)
		default:
			require.NoError(t, err)
			require.Equal(t, lpxCRD.UID, current.UID)
			require.Equal(t, lpxCRD.Spec, current.Spec)
			require.Equal(t, lpxCRD.Annotations, current.Annotations)
		}
	}

	t.Log("Verify installed schemas preserve DGD placement and keep model downloads on the private child")
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
				spec := version.Schema.OpenAPIV3Schema.Properties["spec"]
				roles := spec.Properties["roles"]
				require.Contains(t, roles.Items.Schema.Properties, "replicas")
				require.Contains(t, roles.Items.Schema.Properties, "podTemplate")
				require.NotContains(t, spec.Properties, "lpx")

				templateField := "podTemplate"
				if version.Name == "v1alpha1" {
					templateField = "extraPodSpec"
				}
				template := spec.Properties[templateField]
				if template.Type != objectType || template.Description == "" {
					t.Errorf("%s %s is missing the documented %s schema", name, version.Name, templateField)
				}
			}
			continue
		}
		for _, version := range crd.Spec.Versions {
			status := version.Schema.OpenAPIV3Schema.Properties["status"]
			if name == "dynamographdeployments.nvidia.com" {
				t.Log("Preserve upstream role PodTemplate documentation in both DGD versions")
				spec := version.Schema.OpenAPIV3Schema.Properties["spec"]
				var component *apiextensionsv1.JSONSchemaProps
				typeField := "type"
				if version.Name == "v1alpha1" {
					component = spec.Properties["services"].AdditionalProperties.Schema
					typeField = "componentType"
				} else {
					component = spec.Properties["components"].Items.Schema
				}
				for _, field := range []string{typeField, "lpx"} {
					require.Contains(t, component.Properties[field].Description, "lpx.enabled")
					require.Contains(t, component.Properties[field].Description, "may change incompatibly")
				}
				template := component.Properties["roles"].Items.Schema.Properties["podTemplate"]
				require.NotEmpty(t, template.Description)
				require.NotEmpty(t, template.Properties["spec"].Description)

				if _, exists := status.Properties["modelDownload"]; exists {
					t.Errorf("%s %s retains status.modelDownload", name, version.Name)
				}
				if status.Properties["placement"].Type != objectType {
					t.Errorf("%s %s is missing the established status.placement projection", name, version.Name)
				}
				require.NotContains(t, status.Properties, "lpx")
				continue
			} else if _, exists := status.Properties["lpx"]; exists {
				t.Errorf("%s %s unexpectedly groups private child status", name, version.Name)
			}
			if status.Properties["modelDownload"].Type != objectType {
				t.Errorf("%s %s is missing the typed modelDownload projection", name, version.Name)
			}
		}
	}
}

func TestConfigureGeneratedDGDRV1Beta2CRDs(t *testing.T) {
	tests := []struct {
		file      string
		wantApply bool
	}{
		{file: "../../config/crd/bases/nvidia.com_dynamographdeploymentrequests.yaml", wantApply: true},
		{file: "../../config/crd/bases/nvidia.com_dynamographdeploymentruns.yaml"},
		{file: "../../config/crd/bases/nvidia.com_dynamographdeploymentcandidates.yaml"},
	}

	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			data, err := os.ReadFile(tt.file)
			if err != nil {
				t.Fatalf("read generated CRD: %v", err)
			}
			crd := &apiextensionsv1.CustomResourceDefinition{}
			if err := yaml.Unmarshal(data, crd); err != nil {
				t.Fatalf("decode generated CRD: %v", err)
			}

			if got := configureDGDRV1Beta2CRD(crd); got != tt.wantApply {
				t.Fatalf("configureDGDRV1Beta2CRD() = %v, want %v", got, tt.wantApply)
			}
			if crd.Name == dgdrCRDName {
				for _, version := range crd.Spec.Versions {
					if version.Name == "v1beta2" {
						t.Fatal("generated request CRD still contains v1beta2")
					}
				}
			}
		})
	}
}

func TestGeneratedDGDRV1Beta2RequestSchema(t *testing.T) {
	const crdPath = "../../config/crd/bases/nvidia.com_dynamographdeploymentrequests.yaml"

	data, err := os.ReadFile(crdPath)
	require.NoError(t, err)
	crd := &apiextensionsv1.CustomResourceDefinition{}
	require.NoError(t, yaml.Unmarshal(data, crd))

	var version *apiextensionsv1.CustomResourceDefinitionVersion
	for index := range crd.Spec.Versions {
		if crd.Spec.Versions[index].Name == "v1beta2" {
			version = &crd.Spec.Versions[index]
			break
		}
	}
	require.NotNil(t, version)

	specSchema := version.Schema.OpenAPIV3Schema.Properties["spec"]
	require.Equal(t, []string{"model"}, specSchema.Required)
	for _, removed := range []string{"modelRef", "search", "image", "rerun", "overrides"} {
		require.NotContains(t, specSchema.Properties, removed)
	}
	require.NotContains(t, specSchema.Properties["model"].Properties, "revision")
	require.NotContains(t, specSchema.Properties["workload"].Properties, "trace")

	var internalSchema apiextensions.JSONSchemaProps
	require.NoError(t, apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(
		version.Schema.OpenAPIV3Schema,
		&internalSchema,
		nil,
	))
	schemaValidator, _, err := apiextensionsvalidation.NewSchemaValidator(&internalSchema)
	require.NoError(t, err)
	structural, err := structuralschema.NewStructural(&internalSchema)
	require.NoError(t, err)
	celValidator := apitest.VersionValidatorsFromFile(t, crdPath)["v1beta2"]
	require.NotNil(t, celValidator)

	validate := func(t *testing.T, object map[string]interface{}) []string {
		t.Helper()
		var messages []string
		for _, validationErr := range apiextensionsvalidation.ValidateCustomResource(nil, object, schemaValidator) {
			messages = append(messages, validationErr.Error())
		}
		for _, validationErr := range celValidator(object, nil) {
			messages = append(messages, validationErr.Error())
		}
		for _, validationErr := range listtype.ValidateListSetsAndMaps(field.NewPath(""), structural, object) {
			messages = append(messages, validationErr.Error())
		}
		return messages
	}

	t.Run("defaults", func(t *testing.T) {
		object := validDGDRV1Beta2Request()
		spec := object["spec"].(map[string]interface{})
		delete(spec, "backends")
		delete(spec["objective"].(map[string]interface{}), "mode")
		delete(spec["objective"].(map[string]interface{}), "sla")
		delete(spec, "recommendation")

		structuraldefaulting.Default(object, structural)

		require.Equal(t, []interface{}{"vllm", "sglang", "trtllm"}, spec["backends"])
		require.Equal(t, "Optimize", spec["objective"].(map[string]interface{})["mode"])
		require.EqualValues(t, 2000, spec["objective"].(map[string]interface{})["sla"].(map[string]interface{})["ttftMs"])
		require.EqualValues(t, 30, spec["objective"].(map[string]interface{})["sla"].(map[string]interface{})["itlMs"])
		require.EqualValues(t, 1, spec["recommendation"].(map[string]interface{})["maxCandidates"])

		optOut := validDGDRV1Beta2Request()
		optOutSLA := requestObjective(optOut)["sla"].(map[string]interface{})
		structuraldefaulting.Default(optOut, structural)
		require.Empty(t, optOutSLA)
	})

	t.Run("model only", func(t *testing.T) {
		object := map[string]interface{}{
			"apiVersion": "nvidia.com/v1beta2",
			"kind":       "DynamoGraphDeploymentRequest",
			"metadata":   map[string]interface{}{"name": "model-only"},
			"spec":       map[string]interface{}{"model": map[string]interface{}{"name": "test/model"}},
		}
		structuraldefaulting.Default(object, structural)
		require.Empty(t, validate(t, object))

		spec := object["spec"].(map[string]interface{})
		require.NotContains(t, spec, "hardware")
		require.NotContains(t, spec, "objective")
		require.NotContains(t, spec, "aisimulate")
		require.Equal(t, []interface{}{"vllm", "sglang", "trtllm"}, spec["backends"])
		require.EqualValues(t, 1, spec["recommendation"].(map[string]interface{})["maxCandidates"])
		requireDefaultOptionAWorkload(t, spec)
	})

	t.Run("empty workload placeholder", func(t *testing.T) {
		object := map[string]interface{}{
			"apiVersion": "nvidia.com/v1beta2",
			"kind":       "DynamoGraphDeploymentRequest",
			"metadata":   map[string]interface{}{"name": "defaultable-workload"},
			"spec": map[string]interface{}{
				"model":    map[string]interface{}{"name": "test/model"},
				"workload": map[string]interface{}{},
			},
		}
		structuraldefaulting.Default(object, structural)
		require.Empty(t, validate(t, object))
		requireDefaultOptionAWorkload(t, object["spec"].(map[string]interface{}))
	})

	t.Run("explicit empty static opts out", func(t *testing.T) {
		object := map[string]interface{}{
			"apiVersion": "nvidia.com/v1beta2",
			"kind":       "DynamoGraphDeploymentRequest",
			"metadata":   map[string]interface{}{"name": "native-workload"},
			"spec": map[string]interface{}{
				"model": map[string]interface{}{"name": "test/model"},
				"workload": map[string]interface{}{
					"static": map[string]interface{}{},
				},
			},
		}
		static := requestStaticWorkload(object)
		structuraldefaulting.Default(object, structural)
		require.Empty(t, validate(t, object))
		require.Empty(t, static)
	})

	tests := []struct {
		name    string
		mutate  func(map[string]interface{})
		wantErr bool
	}{
		{
			name: "ratio below one",
			mutate: func(object map[string]interface{}) {
				requestStaticWorkload(object)["numRequestRatio"] = 0.5
			},
		},
		{
			name: "ttft only SLA",
			mutate: func(object map[string]interface{}) {
				requestObjective(object)["sla"] = map[string]interface{}{"ttftMs": 100.0}
			},
		},
		{
			name: "itl only SLA",
			mutate: func(object map[string]interface{}) {
				requestObjective(object)["sla"] = map[string]interface{}{"itlMs": 10.0}
			},
		},
		{
			name: "e2e only SLA",
			mutate: func(object map[string]interface{}) {
				requestObjective(object)["sla"] = map[string]interface{}{"e2eMs": 100.0}
			},
		},
		{
			name: "empty SLA opt out",
			mutate: func(object map[string]interface{}) {
				requestObjective(object)["sla"] = map[string]interface{}{}
			},
		},
		{
			name: "pareto metrics",
			mutate: func(object map[string]interface{}) {
				objective := requestObjective(object)
				objective["mode"] = "Pareto"
				delete(objective, "metric")
				objective["metrics"] = []interface{}{"ThroughputPerGPU", "TTFT"}
			},
		},
		{
			name: "duplicate pareto metrics",
			mutate: func(object map[string]interface{}) {
				objective := requestObjective(object)
				objective["mode"] = "Pareto"
				delete(objective, "metric")
				objective["metrics"] = []interface{}{"TTFT", "TTFT"}
			},
			wantErr: true,
		},
		{
			name: "omitted request ratio uses underlay default",
			mutate: func(object map[string]interface{}) {
				static := requestStaticWorkload(object)
				delete(static, "numRequestRatio")
			},
		},
		{
			name: "KV load ratio below range",
			mutate: func(object map[string]interface{}) {
				static := requestStaticWorkload(object)
				delete(static, "concurrency")
				static["kvLoadRatio"] = []interface{}{-0.1}
			},
			wantErr: true,
		},
		{
			name: "KV load ratio above range",
			mutate: func(object map[string]interface{}) {
				static := requestStaticWorkload(object)
				delete(static, "concurrency")
				static["kvLoadRatio"] = []interface{}{1.1}
			},
			wantErr: true,
		},
		{
			name: "zero request ratio",
			mutate: func(object map[string]interface{}) {
				requestStaticWorkload(object)["numRequestRatio"] = 0.0
			},
			wantErr: true,
		},
		{
			name: "mixed e2e and token SLA",
			mutate: func(object map[string]interface{}) {
				requestObjective(object)["sla"] = map[string]interface{}{"e2eMs": 100.0, "ttftMs": 50.0}
			},
			wantErr: true,
		},
		{
			name: "goodput without SLA",
			mutate: func(object map[string]interface{}) {
				objective := requestObjective(object)
				objective["metric"] = "Goodput"
				objective["sla"] = map[string]interface{}{}
			},
			wantErr: true,
		},
		{
			name: "legacy metric casing",
			mutate: func(object map[string]interface{}) {
				requestObjective(object)["metric"] = "throughputPerGpu"
			},
			wantErr: true,
		},
		{
			name: "recommendation limit",
			mutate: func(object map[string]interface{}) {
				object["spec"].(map[string]interface{})["recommendation"] = map[string]interface{}{"maxCandidates": int64(101)}
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			object := validDGDRV1Beta2Request()
			tt.mutate(object)
			messages := validate(t, object)
			if tt.wantErr {
				require.NotEmpty(t, messages)
				return
			}
			require.Empty(t, messages)
		})
	}
}

func TestGeneratedDGDRV1Beta2ResultSchemas(t *testing.T) {
	tests := []struct {
		path  string
		check func(*testing.T, apiextensionsv1.JSONSchemaProps)
	}{
		{
			path: "../../config/crd/bases/nvidia.com_dynamographdeploymentcandidates.yaml",
			check: func(t *testing.T, root apiextensionsv1.JSONSchemaProps) {
				spec := root.Properties["spec"]
				status := root.Properties["status"]
				require.Contains(t, spec.Required, "parameters")
				require.Contains(t, spec.Properties, "parameters")
				require.Contains(t, status.Properties, "metrics")
				require.NotContains(t, status.Properties, "conditions")
				require.NotContains(t, status.Properties, "rank")
				require.NotContains(t, status.Properties, "experimental")
			},
		},
		{
			path: "../../config/crd/bases/nvidia.com_dynamographdeploymentruns.yaml",
			check: func(t *testing.T, root apiextensionsv1.JSONSchemaProps) {
				status := root.Properties["status"]
				require.Contains(t, status.Properties, "candidateRefs")
				require.NotContains(t, status.Properties, "provenance")
			},
		},
	}

	for _, tt := range tests {
		t.Run(filepath.Base(tt.path), func(t *testing.T) {
			data, err := os.ReadFile(tt.path)
			require.NoError(t, err)
			crd := &apiextensionsv1.CustomResourceDefinition{}
			require.NoError(t, yaml.Unmarshal(data, crd))
			require.Len(t, crd.Spec.Versions, 1)
			require.Equal(t, "v1beta2", crd.Spec.Versions[0].Name)
			tt.check(t, *crd.Spec.Versions[0].Schema.OpenAPIV3Schema)
		})
	}
}

func validDGDRV1Beta2Request() map[string]interface{} {
	return map[string]interface{}{
		"apiVersion": "nvidia.com/v1beta2",
		"kind":       "DynamoGraphDeploymentRequest",
		"metadata": map[string]interface{}{
			"name": "schema-test",
		},
		"spec": map[string]interface{}{
			"model": map[string]interface{}{"name": "test/model"},
			"backends": []interface{}{
				"vllm",
			},
			"hardware": map[string]interface{}{
				"gpu": map[string]interface{}{
					"sku":    "h100",
					"budget": int64(8),
				},
			},
			"workload": map[string]interface{}{
				"static": map[string]interface{}{
					"isl":             int64(1024),
					"osl":             int64(128),
					"numRequestRatio": 1.0,
					"concurrency":     int64(1),
				},
			},
			"objective": map[string]interface{}{
				"mode":   "Optimize",
				"metric": "ThroughputPerGPU",
				"sla":    map[string]interface{}{},
			},
			"recommendation": map[string]interface{}{
				"maxCandidates": int64(1),
			},
		},
	}
}

func requestObjective(object map[string]interface{}) map[string]interface{} {
	return object["spec"].(map[string]interface{})["objective"].(map[string]interface{})
}

func requestStaticWorkload(object map[string]interface{}) map[string]interface{} {
	return object["spec"].(map[string]interface{})["workload"].(map[string]interface{})["static"].(map[string]interface{})
}

func requireDefaultOptionAWorkload(t *testing.T, spec map[string]interface{}) {
	t.Helper()
	static := spec["workload"].(map[string]interface{})["static"].(map[string]interface{})
	require.EqualValues(t, 4000, static["isl"])
	require.EqualValues(t, 1000, static["osl"])
	ratio := static["kvLoadRatio"].([]interface{})
	require.Len(t, ratio, 2)
	require.EqualValues(t, 0, ratio[0])
	require.EqualValues(t, 1, ratio[1])
}

func TestConfigureDGDRV1Beta2CRD(t *testing.T) {
	tests := []struct {
		name         string
		crdName      string
		wantApply    bool
		wantVersions []string
	}{
		{
			name:         "request retains stable versions",
			crdName:      dgdrCRDName,
			wantApply:    true,
			wantVersions: []string{"v1alpha1", "v1beta1"},
		},
		{
			name:         "run is skipped",
			crdName:      dgdrRunCRDName,
			wantVersions: []string{"v1alpha1", "v1beta1", "v1beta2"},
		},
		{
			name:         "candidate is skipped",
			crdName:      dgdrCandidateCRD,
			wantVersions: []string{"v1alpha1", "v1beta1", "v1beta2"},
		},
		{
			name:         "unrelated CRD is unchanged",
			crdName:      "dynamographdeployments.nvidia.com",
			wantApply:    true,
			wantVersions: []string{"v1alpha1", "v1beta1", "v1beta2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			crd := &apiextensionsv1.CustomResourceDefinition{}
			crd.Name = tt.crdName
			for _, version := range []string{"v1alpha1", "v1beta1", "v1beta2"} {
				crd.Spec.Versions = append(crd.Spec.Versions, apiextensionsv1.CustomResourceDefinitionVersion{Name: version})
			}

			if got := configureDGDRV1Beta2CRD(crd); got != tt.wantApply {
				t.Fatalf("configureDGDRV1Beta2CRD() = %v, want %v", got, tt.wantApply)
			}
			var versions []string
			for _, version := range crd.Spec.Versions {
				versions = append(versions, version.Name)
			}
			if len(versions) != len(tt.wantVersions) {
				t.Fatalf("versions = %v, want %v", versions, tt.wantVersions)
			}
			for i := range versions {
				if versions[i] != tt.wantVersions[i] {
					t.Fatalf("versions = %v, want %v", versions, tt.wantVersions)
				}
			}
		})
	}
}
