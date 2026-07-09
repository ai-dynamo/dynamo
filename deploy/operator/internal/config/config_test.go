/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package config

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert/yaml"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"

	configapi "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
)

func TestEncode(t *testing.T) {
	testScheme := runtime.NewScheme()
	err := configapi.AddToScheme(testScheme)
	if err != nil {
		t.Fatal(err)
	}

	defaultConfig := &configapi.OperatorConfiguration{}
	testScheme.Default(defaultConfig)

	testcases := []struct {
		name       string
		scheme     *runtime.Scheme
		cfg        *configapi.OperatorConfiguration
		wantResult map[string]any
	}{
		{
			name:   "empty",
			scheme: testScheme,
			cfg:    &configapi.OperatorConfiguration{},
			wantResult: map[string]any{
				"apiVersion": "operator.config.dynamo.nvidia.com/v1alpha1",
				"kind":       "OperatorConfiguration",
				"server": map[string]any{
					"metrics": map[string]any{
						"bindAddress": "",
						"port":        0,
					},
					"healthProbe": map[string]any{
						"bindAddress": "",
						"port":        0,
					},
					"webhook": map[string]any{
						"bindAddress":       "",
						"port":              0,
						"host":              "",
						"certDir":           "",
						"certProvisionMode": "",
						"secretName":        "",
						"serviceName":       "",
					},
				},
				"leaderElection": map[string]any{
					"enabled":   false,
					"id":        "",
					"namespace": "",
				},
				"namespace": map[string]any{
					"restricted": "",
					"scope": map[string]any{
						"leaseDuration":      "0s",
						"leaseRenewInterval": "0s",
					},
				},
				"orchestrators": map[string]any{
					"grove": map[string]any{
						"terminationDelay": "0s",
					},
					"lws":              map[string]any{},
					"kaiScheduler":     map[string]any{},
					"volcanoScheduler": map[string]any{},
				},
				"dra": map[string]any{},
				"infrastructure": map[string]any{
					"natsAddress":        "",
					"etcdAddress":        "",
					"modelExpressURL":    "",
					"prometheusEndpoint": "",
				},
				"ingress": map[string]any{
					"virtualServiceGateway":   "",
					"controllerClassName":     "",
					"controllerTLSSecretName": "",
					"hostSuffix":              "",
				},
				"serviceMesh": map[string]any{
					"provider": "",
				},
				"rbac": map[string]any{
					"plannerClusterRoleName":       "",
					"dgdrProfilingClusterRoleName": "",
					"eppClusterRoleName":           "",
				},
				"mpi": map[string]any{
					"sshSecretName":      "",
					"sshSecretNamespace": "",
				},
				"checkpoint": map[string]any{
					"enabled": false,
					"storage": map[string]any{
						"type": "",
						"pvc": map[string]any{
							"pvcName":          "",
							"basePath":         "",
							"create":           false,
							"size":             "",
							"storageClassName": "",
							"accessMode":       "",
						},
						"s3": map[string]any{
							"uri":                  "",
							"credentialsSecretRef": "",
						},
						"oci": map[string]any{
							"uri":                  "",
							"credentialsSecretRef": "",
						},
					},
				},
				"discovery": map[string]any{
					"backend": "",
				},
				"gpu": map[string]any{},
				"logging": map[string]any{
					"level":  "",
					"format": "",
				},
				"security": map[string]any{
					"enableHTTP2": false,
				},
			},
		},
		{
			name:   "default",
			scheme: testScheme,
			cfg:    defaultConfig,
			wantResult: map[string]any{
				"apiVersion": "operator.config.dynamo.nvidia.com/v1alpha1",
				"kind":       "OperatorConfiguration",
				"server": map[string]any{
					"metrics": map[string]any{
						"bindAddress": "0.0.0.0",
						"port":        8080,
						"secure":      true,
					},
					"healthProbe": map[string]any{
						"bindAddress": "0.0.0.0",
						"port":        8081,
					},
					"webhook": map[string]any{
						"bindAddress":       "",
						"port":              9443,
						"host":              "0.0.0.0",
						"certDir":           "/tmp/k8s-webhook-server/serving-certs",
						"certProvisionMode": "auto",
						"secretName":        "webhook-server-cert",
						"serviceName":       "",
					},
				},
				"leaderElection": map[string]any{
					"enabled":   false,
					"id":        "",
					"namespace": "",
				},
				"namespace": map[string]any{
					"restricted": "",
					"scope": map[string]any{
						"leaseDuration":      "30s",
						"leaseRenewInterval": "10s",
					},
				},
				"orchestrators": map[string]any{
					"grove": map[string]any{
						"terminationDelay": "15m0s",
					},
					"lws":              map[string]any{},
					"kaiScheduler":     map[string]any{},
					"volcanoScheduler": map[string]any{},
				},
				"dra": map[string]any{},
				"infrastructure": map[string]any{
					"natsAddress":        "",
					"etcdAddress":        "",
					"modelExpressURL":    "",
					"prometheusEndpoint": "",
				},
				"ingress": map[string]any{
					"virtualServiceGateway":   "",
					"controllerClassName":     "",
					"controllerTLSSecretName": "",
					"hostSuffix":              "",
				},
				"serviceMesh": map[string]any{
					"provider": "",
				},
				"rbac": map[string]any{
					"plannerClusterRoleName":       "",
					"dgdrProfilingClusterRoleName": "",
					"eppClusterRoleName":           "",
				},
				"mpi": map[string]any{
					"sshSecretName":      "",
					"sshSecretNamespace": "",
				},
				"checkpoint": map[string]any{
					"enabled": false,
					"storage": map[string]any{
						"type": "",
						"pvc": map[string]any{
							"pvcName":          "",
							"basePath":         "",
							"create":           false,
							"size":             "",
							"storageClassName": "",
							"accessMode":       "",
						},
						"s3": map[string]any{
							"uri":                  "",
							"credentialsSecretRef": "",
						},
						"oci": map[string]any{
							"uri":                  "",
							"credentialsSecretRef": "",
						},
					},
				},
				"discovery": map[string]any{
					"backend": "kubernetes",
				},
				"gpu": map[string]any{
					"discoveryEnabled": true,
				},
				"logging": map[string]any{
					"level":  "info",
					"format": "json",
				},
				"security": map[string]any{
					"enableHTTP2": false,
				},
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Encode(tc.scheme, tc.cfg)
			if err != nil {
				t.Errorf("Unexpected error:%s", err)
			}
			gotMap := map[string]any{}
			err = yaml.Unmarshal([]byte(got), &gotMap)
			if err != nil {
				t.Errorf("Unable to unmarshal result:%s", err)
			}
			if diff := cmp.Diff(tc.wantResult, gotMap); diff != "" {
				t.Errorf("Unexpected result (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLoad(t *testing.T) {
	testScheme := runtime.NewScheme()
	err := configapi.AddToScheme(testScheme)
	if err != nil {
		t.Fatal(err)
	}

	// TODO: add test cases.
	testcases := []struct {
		name              string
		configFile        string
		wantConfiguration configapi.OperatorConfiguration
		wantOptions       ctrl.Options
		wantError         error
	}{}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			options, cfg, err := Load(testScheme, tc.configFile)
			if tc.wantError == nil {
				if err != nil {
					t.Errorf("Unexpected error:%s", err)
				}
				if diff := cmp.Diff(tc.wantConfiguration, cfg); diff != "" {
					t.Errorf("Unexpected config (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(tc.wantOptions, options); diff != "" {
					t.Errorf("Unexpected options (-want +got):\n%s", diff)
				}
			} else {
				if diff := cmp.Diff(tc.wantError.Error(), err.Error()); diff != "" {
					t.Errorf("Unexpected error (-want +got):\n%s", diff)
				}
			}
		})
	}
}
