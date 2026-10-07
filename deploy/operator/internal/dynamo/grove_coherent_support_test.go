// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	grovecrds "github.com/ai-dynamo/grove/operator/api/core/v1alpha1/crds"
	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/yaml"
)

func TestGroveCoherentSupport(t *testing.T) {
	for _, test := range []struct {
		name, version                         string
		served, coherent, denied, wantSupport bool
	}{
		{name: "old CRD", version: "v1alpha1", served: true},
		{name: "supported CRD", version: "v1alpha1", served: true, coherent: true, wantSupport: true},
		{name: "unserved version", version: "v1alpha1", coherent: true},
		{name: "different version", version: "v1beta1", served: true, coherent: true},
		{name: "read forbidden", version: "v1alpha1", served: true, coherent: true, denied: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Install a PCS schema with the selected strategy capability")
			scheme := runtime.NewScheme()
			require.NoError(t, apiextensionsv1.AddToScheme(scheme))
			values := []apiextensionsv1.JSON{{Raw: []byte(`"RollingRecreate"`)}, {Raw: []byte(`"OnDelete"`)}}
			if test.coherent {
				values = append(values, apiextensionsv1.JSON{Raw: []byte(`"Coherent"`)})
			}
			crd := &apiextensionsv1.CustomResourceDefinition{ObjectMeta: metav1.ObjectMeta{Name: "podcliquesets.grove.io"}, Spec: apiextensionsv1.CustomResourceDefinitionSpec{Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{Name: test.version, Served: test.served, Schema: &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Properties: map[string]apiextensionsv1.JSONSchemaProps{
				"spec": {Properties: map[string]apiextensionsv1.JSONSchemaProps{"updateStrategy": {Properties: map[string]apiextensionsv1.JSONSchemaProps{"type": {Enum: values}}}}},
			}}}}}}}
			builder := fake.NewClientBuilder().WithScheme(scheme).WithObjects(crd)
			if test.denied {
				builder.WithInterceptorFuncs(interceptor.Funcs{Get: func(context.Context, client.WithWatch, client.ObjectKey, client.Object, ...client.GetOption) error {
					return apierrors.NewForbidden(schema.GroupResource{Group: "apiextensions.k8s.io", Resource: "customresourcedefinitions"}, crd.Name, errors.New("RBAC denied"))
				}})
			}
			reader := builder.Build()
			support := NewGroveCoherentSupport(reader, reader)

			t.Log("Reject unsupported coherent intent without hiding permission errors")
			err := support.Check(t.Context())
			if test.denied {
				require.True(t, apierrors.IsForbidden(err))
				require.NotErrorIs(t, err, ErrGroveCoherentUnsupported)
			} else if test.wantSupport {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, ErrGroveCoherentUnsupported)
			}

		})
	}
}

func TestGroveCoherentSupportCachesRevisions(t *testing.T) {
	for _, initiallySupported := range []bool{false, true} {
		t.Run(fmt.Sprintf("initiallySupported=%t", initiallySupported), func(t *testing.T) {
			t.Log("Keep CRD metadata and the live schema behind separate readers")
			scheme := runtime.NewScheme()
			require.NoError(t, apiextensionsv1.AddToScheme(scheme))
			crd := newGroveCoherentSupportTestCRD(t, initiallySupported)
			metadataClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(crd).Build()
			metadataReader := interceptor.NewClient(metadataClient, interceptor.Funcs{
				Get: func(ctx context.Context, reader client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
					require.IsType(t, &metav1.PartialObjectMetadata{}, object, "cached reads must not create a full-object CRD informer")
					return reader.Get(ctx, key, object, opts...)
				},
			})
			schemaReads := 0
			schemaClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(crd).Build()
			apiReader := interceptor.NewClient(schemaClient, interceptor.Funcs{
				Get: func(ctx context.Context, reader client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
					require.IsType(t, &apiextensionsv1.CustomResourceDefinition{}, object)
					schemaReads++
					return reader.Get(ctx, key, object, opts...)
				},
			})
			support := NewGroveCoherentSupport(metadataReader, apiReader)

			t.Log("Reuse both positive and negative capability results for an unchanged revision")
			for range 2 {
				err := support.Check(t.Context())
				if initiallySupported {
					require.NoError(t, err)
				} else {
					require.ErrorIs(t, err, ErrGroveCoherentUnsupported)
				}
			}
			require.Equal(t, 1, schemaReads)

			t.Log("Changing the CRD revision refreshes the capability without restarting the checker")
			changed := newGroveCoherentSupportTestCRD(t, !initiallySupported)
			metadataChange := changed.DeepCopy()
			require.NoError(t, schemaClient.Update(t.Context(), changed))
			require.NoError(t, metadataClient.Update(t.Context(), metadataChange))
			err := support.Check(t.Context())
			if initiallySupported {
				require.ErrorIs(t, err, ErrGroveCoherentUnsupported)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, 2, schemaReads)

			t.Log("Recreating the CRD invalidates its capability even if the resource version matches")
			require.NoError(t, schemaClient.Delete(t.Context(), changed))
			require.NoError(t, metadataClient.Delete(t.Context(), changed))
			replacement := newGroveCoherentSupportTestCRD(t, initiallySupported)
			replacement.UID = "replacement-crd"
			replacement.ResourceVersion = ""
			metadataReplacement := replacement.DeepCopy()
			require.NoError(t, schemaClient.Create(t.Context(), replacement))
			require.NoError(t, metadataClient.Create(t.Context(), metadataReplacement))
			require.NoError(t, schemaClient.Update(t.Context(), replacement))
			require.NoError(t, metadataClient.Update(t.Context(), metadataReplacement))
			require.Equal(t, changed.ResourceVersion, replacement.ResourceVersion)
			err = support.Check(t.Context())
			if initiallySupported {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, ErrGroveCoherentUnsupported)
			}
			require.Equal(t, 3, schemaReads)
		})
	}
}

func TestGroveCoherentSupportUsesFetchedRevision(t *testing.T) {
	t.Log("Observe metadata behind the live CRD schema revision")
	scheme := runtime.NewScheme()
	require.NoError(t, apiextensionsv1.AddToScheme(scheme))
	old := newGroveCoherentSupportTestCRD(t, false)
	current := newGroveCoherentSupportTestCRD(t, true)
	current.ResourceVersion = "2"
	metadataClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(old).Build()
	schemaClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(current).Build()
	schemaReads := 0
	apiReader := interceptor.NewClient(schemaClient, interceptor.Funcs{
		Get: func(ctx context.Context, reader client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			schemaReads++
			return reader.Get(ctx, key, object, opts...)
		},
	})
	support := NewGroveCoherentSupport(metadataClient, apiReader)
	require.NoError(t, support.Check(t.Context()))
	require.Equal(t, 1, schemaReads)

	t.Log("When the informer catches up, reuse the result obtained from that schema revision")
	require.NoError(t, metadataClient.Update(t.Context(), old))
	require.Equal(t, current.ResourceVersion, old.ResourceVersion)
	require.NoError(t, support.Check(t.Context()))
	require.Equal(t, 1, schemaReads)
}

func TestGroveCoherentSupportRetriesReadFailures(t *testing.T) {
	for _, failMetadata := range []bool{false, true} {
		t.Run(fmt.Sprintf("metadata=%t", failMetadata), func(t *testing.T) {
			t.Log("Fail one capability read without turning it into a cached unsupported result")
			scheme := runtime.NewScheme()
			require.NoError(t, apiextensionsv1.AddToScheme(scheme))
			crd := newGroveCoherentSupportTestCRD(t, true)
			kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(crd).Build()
			attempts := 0
			failed := interceptor.NewClient(kube, interceptor.Funcs{
				Get: func(ctx context.Context, reader client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
					attempts++
					if attempts == 1 {
						return apierrors.NewForbidden(schema.GroupResource{Group: apiextensionsv1.GroupName, Resource: "customresourcedefinitions"}, key.Name, errors.New("temporary denial"))
					}
					return reader.Get(ctx, key, object, opts...)
				},
			})
			metadataReader, apiReader := client.Reader(kube), client.Reader(failed)
			if failMetadata {
				metadataReader, apiReader = failed, kube
			}
			support := NewGroveCoherentSupport(metadataReader, apiReader)
			err := support.Check(t.Context())
			require.True(t, apierrors.IsForbidden(err))
			require.NotErrorIs(t, err, ErrGroveCoherentUnsupported)

			t.Log("Retry the same revision successfully after the transient read failure")
			require.NoError(t, support.Check(t.Context()))
			require.Equal(t, 2, attempts)
		})
	}
}

func TestGroveCoherentSupportConcurrentChecksShareSchemaRead(t *testing.T) {
	t.Log("Share a capability checker across concurrent workload reconciliations")
	scheme := runtime.NewScheme()
	require.NoError(t, apiextensionsv1.AddToScheme(scheme))
	crd := newGroveCoherentSupportTestCRD(t, true)
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(crd).Build()
	var schemaReads atomic.Int32
	apiReader := interceptor.NewClient(kube, interceptor.Funcs{
		Get: func(ctx context.Context, reader client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			schemaReads.Add(1)
			return reader.Get(ctx, key, object, opts...)
		},
	})
	support := NewGroveCoherentSupport(kube, apiReader)
	results := make(chan error, 12)
	for range cap(results) {
		go func() {
			results <- support.Check(t.Context())
		}()
	}

	t.Log("Every reconciliation succeeds with one schema fetch for the shared revision")
	for range cap(results) {
		require.NoError(t, <-results)
	}
	require.Equal(t, int32(1), schemaReads.Load())
}

func newGroveCoherentSupportTestCRD(t *testing.T, coherent bool) *apiextensionsv1.CustomResourceDefinition {
	t.Helper()
	crd := &apiextensionsv1.CustomResourceDefinition{}
	require.NoError(t, yaml.Unmarshal([]byte(grovecrds.PodCliqueSetCRD()), crd))
	crd.UID = "original-crd"
	crd.ResourceVersion = "1"
	if !coherent {
		for i := range crd.Spec.Versions {
			properties := crd.Spec.Versions[i].Schema.OpenAPIV3Schema.Properties["spec"].Properties["updateStrategy"].Properties
			strategy := properties["type"]
			strategy.Enum = []apiextensionsv1.JSON{{Raw: []byte("\"RollingRecreate\"")}, {Raw: []byte("\"OnDelete\"")}}
			properties["type"] = strategy
		}
	}
	return crd
}
