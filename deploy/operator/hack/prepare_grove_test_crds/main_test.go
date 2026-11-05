// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"testing"

	grovecrds "github.com/ai-dynamo/grove/operator/api/core/crds"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/yaml"
)

func TestCRDsPreservePinnedAlphaSchema(t *testing.T) {
	t.Log("Generate all five Grove test CRDs from the pinned module")
	crds, err := testCRDs()
	require.NoError(t, err)
	require.Len(t, crds, 5)
	for _, crd := range crds {
		require.Len(t, crd.Spec.Versions, 1, crd.Name)
		require.Equal(t, "v1alpha1", crd.Spec.Versions[0].Name, crd.Name)
		require.True(t, crd.Spec.Versions[0].Served, crd.Name)
		require.True(t, crd.Spec.Versions[0].Storage, crd.Name)
		require.Equal(t, apiextensionsv1.NoneConverter, crd.Spec.Conversion.Strategy, crd.Name)
		require.Nil(t, crd.Spec.Conversion.Webhook, crd.Name)
	}

	t.Log("Check that the PCS schema, defaults, CEL rules, and subresources are unchanged")
	var original apiextensionsv1.CustomResourceDefinition
	require.NoError(t, yaml.Unmarshal([]byte(grovecrds.PodCliqueSetCRD()), &original))
	for _, version := range original.Spec.Versions {
		if version.Name == "v1alpha1" {
			version.Storage = true
			require.Equal(t, version, crds[0].Spec.Versions[0])
		}
	}
}

func TestPodCliqueSetRequestsWithoutGroveConversionService(t *testing.T) {
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		t.Skip("requires local envtest assets")
	}

	t.Log("Install the prepared Grove CRDs into an API server with no Grove operator")
	crds, err := testCRDs()
	require.NoError(t, err)
	env := &envtest.Environment{CRDs: crds}
	config, err := env.Start()
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, env.Stop()) })
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, grovev1alpha1.AddToScheme(scheme))
	c, err := client.New(config, client.Options{Scheme: scheme})
	require.NoError(t, err)
	require.NoError(t, c.Create(t.Context(), &corev1.Namespace{Name: "grove-test"}))

	t.Log("Create the alpha PodCliqueSet used by the cluster-test workload provider")
	pcs := &grovev1alpha1.PodCliqueSet{
		Name: "workers", Namespace: "grove-test",
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Replicas: 1,
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{
					Name: "worker",
					Spec: grovev1alpha1.PodCliqueSpec{
						RoleName: "worker", Replicas: 1, MinAvailable: new(int32(1)),
						PodSpec: corev1.PodSpec{Containers: []corev1.Container{{Name: "worker", Image: "registry.k8s.io/pause:3.10"}}},
					},
				}},
			},
		},
	}
	require.NoError(t, c.Create(t.Context(), pcs))

	t.Log("Read, list, update, and delete without a conversion webhook service")
	observed := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), observed))
	var list grovev1alpha1.PodCliqueSetList
	require.NoError(t, c.List(t.Context(), &list, client.InNamespace(pcs.Namespace)))
	require.Len(t, list.Items, 1)
	observed.Spec.Replicas = 2
	require.NoError(t, c.Update(t.Context(), observed))
	require.NoError(t, c.Delete(t.Context(), observed))
}
