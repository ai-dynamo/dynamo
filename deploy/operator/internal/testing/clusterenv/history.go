// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clusterenv

import (
	"context"
	"errors"
	"fmt"
	"reflect"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	admissionv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	extensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// PrepareHistory must run on an exclusive test cluster before candidate admission
// or Dynamo controllers are active. The returned cleanup is required even when setup fails.
func PrepareHistory(ctx context.Context, c client.Client, snapshots map[string][]unstructured.Unstructured) (func(context.Context) error, error) {
	// Refuse to rewrite storage underneath an installed admission stack.
	var mutating admissionv1.MutatingWebhookConfigurationList
	var validating admissionv1.ValidatingWebhookConfigurationList
	if err := c.List(ctx, &mutating); err != nil {
		return nil, err
	}
	if err := c.List(ctx, &validating); err != nil {
		return nil, err
	}
	if len(mutating.Items)+len(validating.Items) != 0 {
		return nil, fmt.Errorf("history seeding requires a dedicated cluster without installed admission webhooks")
	}
	var crds extensions.CustomResourceDefinitionList
	if err := c.List(ctx, &crds); err != nil {
		return nil, err
	}
	originals := map[string]extensions.CustomResourceDefinitionSpec{}
	namespaces := []string{}
	restoreCRDs := func(ctx context.Context) error {
		errs := make([]error, 0, len(originals))
		for name, spec := range originals {
			// The API server rewrites CRD status after each spec change, which races this read.
			errs = append(errs, retry.RetryOnConflict(retry.DefaultRetry, func() error {
				var current extensions.CustomResourceDefinition
				if err := c.Get(ctx, client.ObjectKey{Name: name}, &current); err != nil {
					return err
				}
				current.Spec = *spec.DeepCopy()
				return c.Update(ctx, &current)
			}))
		}
		return errors.Join(errs...)
	}
	cleanup := func(ctx context.Context) error {
		var errs []error
		// On partial startup there may be no controller available to remove finalizers.
		// Only namespaces successfully created by this invocation are ever touched.
		for _, namespace := range namespaces {
			for _, captured := range snapshots[namespace] {
				object := captured.DeepCopy()
				if err := c.Get(ctx, client.ObjectKeyFromObject(object), object); apierrors.IsNotFound(err) {
					continue
				} else if err != nil {
					errs = append(errs, err)
					continue
				}
				if len(object.GetFinalizers()) > 0 {
					object.SetFinalizers(nil)
					errs = append(errs, c.Update(ctx, object))
				}
			}
			err := c.Delete(ctx, &corev1.Namespace{Name: namespace})
			if !apierrors.IsNotFound(err) {
				errs = append(errs, err)
			}
		}
		errs = append(errs, restoreCRDs(ctx))
		return errors.Join(errs...)
	}
	required := map[string]map[string]bool{}
	for namespace, objects := range snapshots {
		if err := c.Create(ctx, &corev1.Namespace{Name: namespace}); err != nil {
			return cleanup, err
		}
		namespaces = append(namespaces, namespace)

		// Native controllers already run while predecessor workloads are reconstructed.
		if err := blockQuota(ctx, c, workloadBlockQuota(namespace)); err != nil {
			return cleanup, err
		}

		for _, object := range objects {
			gvk := object.GroupVersionKind()
			for _, crd := range crds.Items {
				if crd.Spec.Group == gvk.Group && crd.Spec.Names.Kind == gvk.Kind {
					if required[crd.Name] == nil {
						required[crd.Name] = map[string]bool{}
					}
					required[crd.Name][gvk.Version] = true
				}
			}
		}
	}
	for _, crd := range crds.Items {
		versions, needed := required[crd.Name]
		if !needed {
			continue
		}
		originals[crd.Name] = *crd.Spec.DeepCopy()
		crd.Spec.Conversion = &extensions.CustomResourceConversion{Strategy: extensions.NoneConverter}
		for i := range crd.Spec.Versions {
			v := &crd.Spec.Versions[i]
			v.Schema = &extensions.CustomResourceValidation{OpenAPIV3Schema: &extensions.JSONSchemaProps{Type: "object", XPreserveUnknownFields: new(true)}}
			v.DeprecationWarning = nil
			v.Deprecated = false
			v.Served = true
			delete(versions, v.Name)
		}
		for version := range versions {
			crd.Spec.Versions = append(crd.Spec.Versions, extensions.CustomResourceDefinitionVersion{
				Name: version, Served: true, Schema: &extensions.CustomResourceValidation{OpenAPIV3Schema: &extensions.JSONSchemaProps{Type: "object", XPreserveUnknownFields: new(true)}},
				Subresources: &extensions.CustomResourceSubresources{Status: &extensions.CustomResourceSubresourceStatus{}},
			})
		}
		if err := c.Update(ctx, &crd); err != nil {
			return cleanup, err
		}
	}
	for namespace, objects := range snapshots {
		if err := golden.ApplySnapshot(ctx, c, namespace, objects); err != nil {
			return cleanup, err
		}
		if err := verifyHistoricalObjects(ctx, c, crds.Items, objects); err != nil {
			return cleanup, err
		}
	}
	return cleanup, restoreCRDs(ctx)
}

func verifyHistoricalObjects(ctx context.Context, c client.Reader, crds []extensions.CustomResourceDefinition, objects []unstructured.Unstructured) error {
	// Read back while schemas are still permissive. After restoring production
	// schemas, Kubernetes is allowed to prune unknown fields on reads as well
	// as writes; checking only after takeover cannot prove they reached storage.
	for _, captured := range objects {
		gvk := captured.GroupVersionKind()
		custom := false
		for _, crd := range crds {
			custom = custom || (crd.Spec.Group == gvk.Group && crd.Spec.Names.Kind == gvk.Kind)
		}
		if !custom {
			continue
		}
		stored := captured.DeepCopy()
		if err := c.Get(ctx, client.ObjectKeyFromObject(stored), stored); err != nil {
			return err
		}
		for _, field := range []string{"spec", "status"} {
			if !reflect.DeepEqual(captured.Object[field], stored.Object[field]) {
				return fmt.Errorf("historical %s %s was normalized before takeover: %s", captured.GetKind(), captured.GetName(), field)
			}
		}
	}
	return nil
}
