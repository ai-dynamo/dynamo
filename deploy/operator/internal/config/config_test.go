/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package config

import (
	"errors"
	"io/fs"
	"net"
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert/yaml"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlcache "sigs.k8s.io/controller-runtime/pkg/cache"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

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

	validConfigFile := t.TempDir() + "/config.yaml"
	err = os.WriteFile(validConfigFile, []byte(`apiVersion: operator.config.dynamo.nvidia.com/v1alpha1
kind: OperatorConfiguration
rbac:
  plannerClusterRoleName: planner-role
  dgdrProfilingClusterRoleName: dgdr-profiling-role
  eppClusterRoleName: epp-role
mpi:
  sshSecretName: mpi-ssh
  sshSecretNamespace: default
`), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	defaultCfg := configapi.OperatorConfiguration{
		TypeMeta: metav1.TypeMeta{
			APIVersion: configapi.SchemeGroupVersion.String(),
			Kind:       "OperatorConfiguration",
		},
		Server: configapi.ServerConfiguration{
			Metrics: configapi.MetricsServer{
				Server: configapi.Server{
					BindAddress: "0.0.0.0",
					Port:        8080,
				},
				Secure: ptr.To(true),
			},
			HealthProbe: configapi.Server{
				BindAddress: "0.0.0.0",
				Port:        8081,
			},
			Webhook: configapi.WebhookServer{
				Server: configapi.Server{
					Port: 9443,
				},
				Host:              "0.0.0.0",
				CertDir:           "/tmp/k8s-webhook-server/serving-certs",
				CertProvisionMode: configapi.CertProvisionModeAuto,
				SecretName:        "webhook-server-cert",
			},
		},
		Namespace: configapi.NamespaceConfiguration{
			Scope: configapi.NamespaceScopeConfiguration{
				LeaseDuration:      metav1.Duration{Duration: 30 * time.Second},
				LeaseRenewInterval: metav1.Duration{Duration: 10 * time.Second},
			},
		},
		Orchestrators: configapi.OrchestratorConfiguration{
			Grove: configapi.GroveConfiguration{
				TerminationDelay: metav1.Duration{Duration: 15 * time.Minute},
			},
		},
		Discovery: configapi.DiscoveryConfiguration{
			Backend: configapi.DiscoveryBackendKubernetes,
		},
		GPU: configapi.GPUConfiguration{
			DiscoveryEnabled: ptr.To(true),
		},
		Logging: configapi.LoggingConfiguration{
			Level:  "info",
			Format: "json",
		},
	}

	defaultControlOptions := ctrl.Options{
		Metrics: metricsserver.Options{
			BindAddress:   "0.0.0.0:8080",
			SecureServing: true,
		},
		HealthProbeBindAddress: "0.0.0.0:8081",
	}

	configCmpOpts := []cmp.Option{}

	ctrlOptsCmpOpts := []cmp.Option{
		cmpopts.IgnoreUnexported(ctrl.Options{}),
		cmpopts.IgnoreUnexported(webhook.DefaultServer{}),
		cmpopts.IgnoreUnexported(ctrlcache.Options{}),
		cmpopts.IgnoreUnexported(net.ListenConfig{}),
		cmpopts.IgnoreFields(metricsserver.Options{}, "FilterProvider", "TLSOpts"),
		cmpopts.IgnoreFields(ctrl.Options{}, "Scheme", "Logger", "WebhookServer"),
		cmpopts.IgnoreFields(ctrl.Options{}, "Controller", "Logger"),
	}

	testcases := []struct {
		name              string
		configFile        string
		wantConfiguration configapi.OperatorConfiguration
		wantOptions       ctrl.Options
		wantError         error
	}{
		{
			name:              "default config",
			configFile:        "",
			wantConfiguration: defaultCfg,
			wantOptions:       defaultControlOptions,
			wantError: errors.New("[mpi.sshSecretName: Required value: MPI SSH secret name is required, " +
				"mpi.sshSecretNamespace: Required value: MPI SSH secret namespace is required, " +
				"rbac.plannerClusterRoleName: Required value: planner ClusterRole name is required in cluster-wide mode, " +
				"rbac.dgdrProfilingClusterRoleName: Required value: DGDR profiling ClusterRole name is required in cluster-wide mode, " +
				"rbac.eppClusterRoleName: Required value: EPP ClusterRole name is required in cluster-wide mode]"),
		},
		{
			name:       "minimal valid",
			configFile: validConfigFile,
			wantConfiguration: func() configapi.OperatorConfiguration {
				cfg := (&defaultCfg).DeepCopy()
				cfg.RBAC = configapi.RBACConfiguration{
					PlannerClusterRoleName:       "planner-role",
					DGDRProfilingClusterRoleName: "dgdr-profiling-role",
					EPPClusterRoleName:           "epp-role",
				}
				cfg.MPI = configapi.MPIConfiguration{
					SSHSecretName:      "mpi-ssh",
					SSHSecretNamespace: "default",
				}
				return *cfg
			}(),
			wantOptions: defaultControlOptions,
		},
		{
			name:       "bad path",
			configFile: ".",
			wantError: &fs.PathError{
				Op:   "read",
				Path: ".",
				Err:  errors.New("is a directory"),
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			options, cfg, err := Load(testScheme, tc.configFile)
			if tc.wantError == nil {
				if err != nil {
					t.Errorf("Unexpected error:%s", err)
				}
				if diff := cmp.Diff(tc.wantConfiguration, cfg, configCmpOpts...); diff != "" {
					t.Errorf("Unexpected config (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(tc.wantOptions, options, ctrlOptsCmpOpts...); diff != "" {
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
