//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"testing"
	"time"

	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	grovecrds "github.com/ai-dynamo/grove/operator/api/core/v1alpha1/crds"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	controllerconfig "sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/yaml"
)

func TestGroveCRDWatchesCacheOnlyMetadata(t *testing.T) {
	for _, lpxEnabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("lpx=%t", lpxEnabled), func(t *testing.T) {
			t.Log("Register production Grove watches with implicit informer creation disabled")
			runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LPX: lpxEnabled}}
			env := operatorenv.New(operatorenv.Options{RuntimeConfig: runtimeConfig, SetupWebhooks: setupProductionWebhooks}).RunT(t)

			t.Log("Install the pinned Grove schema and the LPX watch dependency when enabled")
			pcsCRD := &apiextensionsv1.CustomResourceDefinition{}
			require.NoError(t, yaml.Unmarshal([]byte(grovecrds.PodCliqueSetCRD()), pcsCRD))
			crds := []*apiextensionsv1.CustomResourceDefinition{pcsCRD}
			if lpxEnabled {
				require.NoError(t, lpxv1alpha1.AddToScheme(env.Client().Scheme()))
				crds = append(crds, &apiextensionsv1.CustomResourceDefinition{
					ObjectMeta: metav1.ObjectMeta{Name: "lpupipelinerequests." + lpxv1alpha1.APIGroup},
					Spec: apiextensionsv1.CustomResourceDefinitionSpec{
						Group: lpxv1alpha1.APIGroup, Scope: apiextensionsv1.NamespaceScoped,
						Names: apiextensionsv1.CustomResourceDefinitionNames{Kind: "LpuPipelineRequest", ListKind: "LpuPipelineRequestList", Plural: "lpupipelinerequests"},
						Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
							Name: "v1alpha1", Served: true, Storage: true,
							Schema: &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: "object", XPreserveUnknownFields: ptr.To(true)}},
						}},
					},
				})
			}
			_, err := envtest.InstallCRDs(env.RESTConfig(), envtest.CRDInstallOptions{CRDs: crds})
			require.NoError(t, err)
			manager, err := ctrl.NewManager(env.RESTConfig(), ctrl.Options{
				Scheme:     env.Client().Scheme(),
				Metrics:    metricsserver.Options{BindAddress: "0"},
				Controller: controllerconfig.Controller{SkipNameValidation: ptr.To(true)},
				Cache: cache.Options{
					ReaderFailOnMissingInformer: true,
					DefaultNamespaces:           map[string]cache.Config{env.Namespace(): {}},
				},
			})
			require.NoError(t, err)
			config := env.OperatorConfig().DeepCopy()
			config.Namespace.Restricted = env.Namespace()
			require.NoError(t, SetupDynamoGraphDeployment(manager, DynamoGraphDeploymentSetupOptions{
				SetupOptions: SetupOptions{Config: config, RuntimeConfig: runtimeConfig},
			}))

			t.Log("Start the manager and wait for its registered informer caches")
			ctx, cancel := context.WithCancel(t.Context())
			done := make(chan error, 1)
			go func() { done <- manager.Start(ctx) }()
			t.Cleanup(func() {
				cancel()
				require.NoError(t, <-done)
			})
			syncCtx, syncCancel := context.WithTimeout(ctx, 10*time.Second)
			defer syncCancel()
			require.True(t, manager.GetCache().WaitForCacheSync(syncCtx))

			t.Log("Schema discovery succeeds without adding a full-object CRD informer")
			support := dynamo.NewGroveCoherentSupport(manager.GetClient(), manager.GetAPIReader())
			require.EventuallyWithT(t, func(collect *assert.CollectT) {
				require.NoError(collect, support.Check(ctx))
			}, 10*time.Second, 100*time.Millisecond)
			crd := &apiextensionsv1.CustomResourceDefinition{}
			err = manager.GetClient().Get(ctx, client.ObjectKey{Name: dynamo.GrovePodCliqueSetCRDName}, crd)
			var notCached *cache.ErrResourceNotCached
			require.ErrorAs(t, err, &notCached)
		})
	}
}
