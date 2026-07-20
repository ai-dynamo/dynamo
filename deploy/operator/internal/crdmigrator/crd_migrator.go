/*
SPDX-FileCopyrightText: Copyright 2025 The Kubernetes Authors.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Derived from kubernetes-sigs/cluster-api/controllers/crdmigrator at v1.13.3,
commit cf0f6c00fbf7d5c5dbf37bd09554c6389de93861.
*/

// Package crdmigrator contains a controller-runtime-only CRD migrator.
package crdmigrator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

const (
	// ObservedGenerationAnnotation records the CRD generation reconciled by the migrator.
	ObservedGenerationAnnotation = "crd-migration.cluster.x-k8s.io/observed-generation"
	storageMigrationCacheTTL     = time.Hour
)

// Phase identifies a phase of CRD migration.
type Phase string

const (
	// StorageVersionMigrationPhase rewrites objects into the current storage version.
	StorageVersionMigrationPhase Phase = "StorageVersionMigration"
	// CleanupManagedFieldsPhase removes managedFields entries for unserved versions.
	CleanupManagedFieldsPhase Phase = "CleanupManagedFields"
)

// ByObjectConfig configures migration for one registered object type.
type ByObjectConfig struct {
	UseCache                            bool
	UseStatusForStorageVersionMigration bool
}

// +kubebuilder:rbac:groups=apiextensions.k8s.io,resources=customresourcedefinitions,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=apiextensions.k8s.io,resources=customresourcedefinitions/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponentdeployments;dynamographdeployments;dynamographdeploymentrequests;dynamographdeploymentscalingadapters,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponentdeployments/status;dynamographdeployments/status;dynamographdeploymentrequests/status;dynamographdeploymentscalingadapters/status,verbs=get;update;patch

// CRDMigrator migrates the configured CRDs.
type CRDMigrator struct {
	Client    client.Client
	APIReader client.Reader

	SkipCRDMigrationPhases []Phase
	Config                 map[client.Object]ByObjectConfig

	phases          sets.Set[Phase]
	configByCRDName map[string]ByObjectConfig
	migrated        *ttlSet
}

// SetupWithManager registers the migrator with a controller-runtime manager.
func (r *CRDMigrator) SetupWithManager(mgr ctrl.Manager, options controller.Options) error {
	if err := r.setup(mgr.GetScheme()); err != nil {
		return err
	}
	if r.phases.Len() == 0 {
		return nil
	}

	return builder.ControllerManagedBy(mgr).
		For(&apiextensionsv1.CustomResourceDefinition{}, builder.OnlyMetadata,
			builder.WithPredicates(predicate.ResourceVersionChangedPredicate{})).
		Named("crdmigrator").
		WithOptions(options).
		Complete(r)
}

func (r *CRDMigrator) setup(scheme *runtime.Scheme) error {
	if r.Client == nil || r.APIReader == nil || len(r.Config) == 0 {
		return errors.New("Client and APIReader must not be nil and Config must not be empty")
	}

	r.phases = sets.New(StorageVersionMigrationPhase, CleanupManagedFieldsPhase)
	for _, phase := range r.SkipCRDMigrationPhases {
		if !r.phases.Has(phase) {
			return fmt.Errorf("invalid phase %q specified in SkipCRDMigrationPhases", phase)
		}
		r.phases.Delete(phase)
	}

	r.configByCRDName = make(map[string]ByObjectConfig, len(r.Config))
	for obj, cfg := range r.Config {
		gvk, err := apiutil.GVKForObject(obj, scheme)
		if err != nil {
			return fmt.Errorf("get GVK for configured object: %w", err)
		}
		resource, _ := meta.UnsafeGuessKindToResource(gvk.GroupKind().WithVersion(gvk.Version))
		name := resource.Resource + "." + gvk.Group
		if _, exists := r.configByCRDName[name]; exists {
			return fmt.Errorf("duplicate migration configuration for CRD %s", name)
		}
		r.configByCRDName[name] = cfg
	}
	r.migrated = newTTLSet(storageMigrationCacheTTL)
	return nil
}

// Reconcile migrates a configured CRD to its declared storage and served versions.
func (r *CRDMigrator) Reconcile(ctx context.Context, req ctrl.Request) (_ ctrl.Result, reterr error) {
	cfg, ok := r.configByCRDName[req.Name]
	if !ok {
		return ctrl.Result{}, nil
	}

	partial := &metav1.PartialObjectMetadata{}
	partial.SetGroupVersionKind(apiextensionsv1.SchemeGroupVersion.WithKind("CustomResourceDefinition"))
	if err := r.Client.Get(ctx, req.NamespacedName, partial); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	currentGeneration := strconv.FormatInt(partial.Generation, 10)
	if partial.Annotations[ObservedGenerationAnnotation] == currentGeneration {
		return ctrl.Result{}, nil
	}

	crd := &apiextensionsv1.CustomResourceDefinition{}
	if err := r.APIReader.Get(ctx, req.NamespacedName, crd); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	currentGeneration = strconv.FormatInt(crd.Generation, 10)
	storageVersion, err := storageVersionForCRD(crd)
	if err != nil {
		return ctrl.Result{}, err
	}

	defer func() {
		if reterr != nil {
			return
		}
		before := crd.DeepCopy()
		if crd.Annotations == nil {
			crd.Annotations = map[string]string{}
		}
		crd.Annotations[ObservedGenerationAnnotation] = currentGeneration
		if err := r.Client.Patch(ctx, crd, client.MergeFrom(before)); err != nil {
			reterr = errors.Join(reterr, fmt.Errorf("patch CRD %s completion annotation: %w", crd.Name, err))
		}
	}()

	var objects []client.Object
	if (r.phases.Has(StorageVersionMigrationPhase) && storageVersionMigrationRequired(crd, storageVersion)) || r.phases.Has(CleanupManagedFieldsPhase) {
		objects, err = r.listCustomResources(ctx, crd, cfg, storageVersion)
		if err != nil {
			return ctrl.Result{}, err
		}
	}

	if r.phases.Has(StorageVersionMigrationPhase) && storageVersionMigrationRequired(crd, storageVersion) {
		if err := r.migrateStorageVersion(ctx, crd, cfg, objects, storageVersion); err != nil {
			return ctrl.Result{}, err
		}
		before := crd.DeepCopy()
		crd.Status.StoredVersions = []string{storageVersion}
		patch := client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{})
		if err := r.Client.Status().Patch(ctx, crd, patch); err != nil {
			return ctrl.Result{}, fmt.Errorf("patch CRD %s storedVersions: %w", crd.Name, err)
		}
	}

	if r.phases.Has(CleanupManagedFieldsPhase) {
		if err := r.cleanupManagedFields(ctx, crd, objects, cfg, storageVersion); err != nil {
			return ctrl.Result{}, err
		}
	}
	return ctrl.Result{}, nil
}

func storageVersionForCRD(crd *apiextensionsv1.CustomResourceDefinition) (string, error) {
	for _, version := range crd.Spec.Versions {
		if version.Storage {
			return version.Name, nil
		}
	}
	return "", fmt.Errorf("could not find storage version for CRD %s", crd.Name)
}

func storageVersionMigrationRequired(crd *apiextensionsv1.CustomResourceDefinition, storageVersion string) bool {
	return len(crd.Status.StoredVersions) != 1 || crd.Status.StoredVersions[0] != storageVersion
}

func (r *CRDMigrator) listCustomResources(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition, cfg ByObjectConfig, storageVersion string) ([]client.Object, error) {
	gvk := schema.GroupVersionKind{Group: crd.Spec.Group, Version: storageVersion, Kind: crd.Spec.Names.ListKind}
	if cfg.UseCache {
		obj, err := r.Client.Scheme().New(gvk)
		if err != nil {
			return nil, fmt.Errorf("create %s list: %w", crd.Spec.Names.Kind, err)
		}
		list, ok := obj.(client.ObjectList)
		if !ok {
			return nil, fmt.Errorf("%s is not an ObjectList", crd.Spec.Names.ListKind)
		}
		if err := r.Client.List(ctx, list); err != nil {
			return nil, fmt.Errorf("list %s via cache: %w", crd.Spec.Names.Kind, err)
		}
		return extractObjects(list)
	}

	list := &metav1.PartialObjectMetadataList{}
	list.SetGroupVersionKind(gvk)
	var result []client.Object
	for {
		if err := r.APIReader.List(ctx, list, client.Continue(list.Continue), client.Limit(500)); err != nil {
			return nil, fmt.Errorf("list %s via live client: %w", crd.Spec.Names.Kind, err)
		}
		items, err := extractObjects(list)
		if err != nil {
			return nil, err
		}
		result = append(result, items...)
		if list.Continue == "" {
			return result, nil
		}
	}
}

func extractObjects(list client.ObjectList) ([]client.Object, error) {
	items, err := meta.ExtractList(list)
	if err != nil {
		return nil, fmt.Errorf("extract list items: %w", err)
	}
	result := make([]client.Object, 0, len(items))
	for _, item := range items {
		obj, ok := item.(client.Object)
		if !ok {
			return nil, fmt.Errorf("list item %T is not a client.Object", item)
		}
		result = append(result, obj)
	}
	return result, nil
}

func (r *CRDMigrator) migrateStorageVersion(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition, cfg ByObjectConfig, objects []client.Object, storageVersion string) error {
	if len(objects) == 0 {
		return nil
	}
	gvk := schema.GroupVersionKind{Group: crd.Spec.Group, Version: storageVersion, Kind: crd.Spec.Names.Kind}
	ctrl.LoggerFrom(ctx).Info("Running storage version migration", "apiVersion", storageVersion, "objects", len(objects), "kind", gvk.Kind)

	var errs []error
	for _, obj := range objects {
		key := fmt.Sprintf("%s %s %d", gvk.Kind, client.ObjectKeyFromObject(obj), crd.Generation)
		if r.migrated.Has(key) {
			r.migrated.Add(key)
			continue
		}
		u := &unstructured.Unstructured{}
		u.SetGroupVersionKind(gvk)
		u.SetNamespace(obj.GetNamespace())
		u.SetName(obj.GetName())
		u.SetUID(obj.GetUID())
		u.SetResourceVersion(obj.GetResourceVersion())

		var err error
		if cfg.UseStatusForStorageVersionMigration {
			err = r.Client.Status().Apply(ctx, client.ApplyConfigurationFromUnstructured(u), client.FieldOwner("crdmigrator"))
		} else {
			err = r.Client.Apply(ctx, client.ApplyConfigurationFromUnstructured(u), client.FieldOwner("crdmigrator"))
		}
		if err != nil && !apierrors.IsNotFound(err) && !apierrors.IsConflict(err) {
			errs = append(errs, fmt.Errorf("%s: %w", klog.KObj(u), err))
			continue
		}
		r.migrated.Add(key)
	}
	if err := errors.Join(errs...); err != nil {
		return fmt.Errorf("migrate storage version of %s objects: %w", gvk.Kind, err)
	}
	return nil
}

func (r *CRDMigrator) cleanupManagedFields(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition, objects []client.Object, cfg ByObjectConfig, storageVersion string) error {
	served := sets.New[string]()
	for _, version := range crd.Spec.Versions {
		if version.Served {
			served.Insert(crd.Spec.Group + "/" + version.Name)
		}
	}

	var errs []error
	for _, obj := range objects {
		if len(obj.GetManagedFields()) == 0 {
			continue
		}
		var getErr error
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			getErr = nil
			managedFields, removed := filterManagedFields(obj, served)
			if !removed {
				return nil
			}
			if len(managedFields) == 0 {
				fields, err := json.Marshal(map[string]any{"f:metadata": map[string]any{"f:name": map[string]any{}}})
				if err != nil {
					return fmt.Errorf("create seeding managedFields entry: %w", err)
				}
				first := obj.GetManagedFields()[0]
				managedFields = append(managedFields, metav1.ManagedFieldsEntry{
					Manager: first.Manager, Operation: first.Operation,
					APIVersion: crd.Spec.Group + "/" + storageVersion,
					Time:       ptr.To(metav1.Now()), FieldsType: "FieldsV1",
					FieldsV1: &metav1.FieldsV1{Raw: fields},
				})
			}
			patch, err := json.Marshal([]map[string]any{
				{"op": "replace", "path": "/metadata/managedFields", "value": managedFields},
				{"op": "replace", "path": "/metadata/resourceVersion", "value": obj.GetResourceVersion()},
			})
			if err != nil {
				return fmt.Errorf("marshal managedFields patch: %w", err)
			}
			err = r.Client.Patch(ctx, obj, client.RawPatch(types.JSONPatchType, patch))
			if err == nil || apierrors.IsNotFound(err) {
				return nil
			}
			if apierrors.IsConflict(err) {
				if cfg.UseCache {
					getErr = r.Client.Get(ctx, client.ObjectKeyFromObject(obj), obj)
				} else {
					getErr = r.APIReader.Get(ctx, client.ObjectKeyFromObject(obj), obj)
				}
			}
			return err
		})
		if err != nil {
			errs = append(errs, errors.Join(fmt.Errorf("%s: %w", klog.KObj(obj), err), getErr))
		}
	}
	if err := errors.Join(errs...); err != nil {
		return fmt.Errorf("clean managedFields of %s objects: %w", crd.Spec.Names.Kind, err)
	}
	return nil
}

func filterManagedFields(obj client.Object, served sets.Set[string]) ([]metav1.ManagedFieldsEntry, bool) {
	filtered := make([]metav1.ManagedFieldsEntry, 0, len(obj.GetManagedFields()))
	removed := false
	for _, entry := range obj.GetManagedFields() {
		if served.Has(entry.APIVersion) {
			filtered = append(filtered, entry)
		} else {
			removed = true
		}
	}
	return filtered, removed
}

type ttlSet struct {
	mu      sync.Mutex
	ttl     time.Duration
	entries map[string]time.Time
}

func newTTLSet(ttl time.Duration) *ttlSet {
	return &ttlSet{ttl: ttl, entries: map[string]time.Time{}}
}

func (s *ttlSet) Add(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries[key] = time.Now().Add(s.ttl)
}

func (s *ttlSet) Has(key string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	expires, ok := s.entries[key]
	if !ok {
		return false
	}
	if time.Now().After(expires) {
		delete(s.entries, key)
		return false
	}
	return true
}
