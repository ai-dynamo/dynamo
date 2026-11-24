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

package controller_common

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strconv"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/events"
	"k8s.io/client-go/util/csaupgrade"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
)

const (
	// NvidiaAnnotationHashKey indicates annotation name for last applied hash by the operator
	NvidiaAnnotationHashKey = "nvidia.com/last-applied-hash"
	// NvidiaAnnotationGenerationKey indicates annotation name for last applied generation by the operator
	// This is used to detect manual changes to resources
	NvidiaAnnotationGenerationKey = "nvidia.com/last-applied-generation"
	// EventReasonOwnershipConflict identifies a parent resource with a child-resource ownership collision.
	EventReasonOwnershipConflict = "OwnershipConflict"
)

type OwnershipConflictError struct {
	Cause error
}

func (e *OwnershipConflictError) Error() string {
	return e.Cause.Error()
}

func (e *OwnershipConflictError) Unwrap() error {
	return e.Cause
}

type Reconciler interface {
	client.Client
	GetRecorder() events.EventRecorder
}

// ResourceGenerator is a function that generates a resource.
// it must return the resource, a boolean indicating if the resource should be deleted, and an error
// if the resource should be deleted, the returned resource must contain the necessary information to delete it (name and namespace)
type ResourceGenerator[T client.Object] func(ctx context.Context) (T, bool, error)

// SyncOption configures resource synchronization.
type SyncOption func(*syncOptions)

type syncOptions struct {
	sharedOwnership   bool
	preserveListOrder bool
}

// WithPreservedListOrder makes authored list ordering significant in spec hashes.
func WithPreservedListOrder() SyncOption {
	return func(options *syncOptions) {
		options.preserveListOrder = true
	}
}

// WithSharedOwnership permits a caller to reconcile a resource whose controller
// owner is another resource. Callers must use this only for a documented,
// intentional shared-resource lifecycle.
func WithSharedOwnership() SyncOption {
	return func(options *syncOptions) {
		options.sharedOwnership = true
	}
}

func resolveSyncOptions(opts []SyncOption) syncOptions {
	var options syncOptions
	for _, opt := range opts {
		opt(&options)
	}
	return options
}

// CheckControllerOwnership verifies that existing is controlled by parentResource.
// A namespaced name is not sufficient evidence of ownership: a resource with no
// controller owner or one owned by a different parent is a collision.
func CheckControllerOwnership(existing, parentResource client.Object, scheme *runtime.Scheme) error {
	if parentResource == nil {
		return nil
	}

	existingOwner := metav1.GetControllerOf(existing)
	if existingOwner == nil {
		return &OwnershipConflictError{Cause: fmt.Errorf(
			"%T %s/%s has no controller owner; refusing to reconcile it for %T %s/%s",
			existing,
			existing.GetNamespace(),
			existing.GetName(),
			parentResource,
			parentResource.GetNamespace(),
			parentResource.GetName(),
		)}
	}

	parentGVK, err := apiutil.GVKForObject(parentResource, scheme)
	if err != nil {
		return fmt.Errorf("get parent GVK: %w", err)
	}
	existingOwnerGV, err := schema.ParseGroupVersion(existingOwner.APIVersion)
	if err != nil ||
		existingOwnerGV.Group != parentGVK.Group ||
		existingOwner.Kind != parentGVK.Kind ||
		existingOwner.Name != parentResource.GetName() ||
		existingOwner.UID != parentResource.GetUID() {
		return &OwnershipConflictError{Cause: &controllerutil.AlreadyOwnedError{Object: existing, Owner: *existingOwner}}
	}

	return nil
}

// SyncResource synchronizes a generated resource with the API server. parentResource may be nil;
// when it is nil, SyncResource neither sets nor validates controller ownership.
//
//nolint:nakedret
func SyncResource[T client.Object](ctx context.Context, r Reconciler, parentResource client.Object, generateResource ResourceGenerator[T], opts ...SyncOption) (modified bool, res T, err error) {
	logs := log.FromContext(ctx)
	options := resolveSyncOptions(opts)

	resource, toDelete, err := generateResource(ctx)
	if err != nil {
		return
	}
	resourceNamespace := resource.GetNamespace()
	resourceName := resource.GetName()
	resourceType := reflect.TypeOf(resource).Elem().Name()
	logs = logs.WithValues("namespace", resourceNamespace, "resourceName", resourceName, "resourceType", resourceType)

	// Retrieve the GroupVersionKind (GVK) of the desired object
	gvk, err := apiutil.GVKForObject(resource, r.Scheme())
	if err != nil {
		logs.Error(err, "Failed to get GVK for object")
		return
	}

	// Create a new instance of the object
	obj, err := r.Scheme().New(gvk)
	if err != nil {
		logs.Error(err, "Failed to create a new object for GVK")
		return
	}

	// Type assertion to ensure the object implements client.Object
	oldResource, ok := obj.(T)
	if !ok {
		return
	}

	err = r.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: resourceNamespace}, oldResource)
	oldResourceIsNotFound := errors.IsNotFound(err)
	if err != nil && !oldResourceIsNotFound {
		r.GetRecorder().Eventf(resource, nil, corev1.EventTypeWarning, fmt.Sprintf("Get%s", resourceType), "Get", "Failed to get %s %s: %s", resourceType, resourceNamespace, err)
		logs.Error(err, "Failed to get resource.")
		return
	}
	err = nil

	if oldResourceIsNotFound {
		if toDelete {
			logs.Info("Resource not found. Nothing to do.")
			return
		}
		logs.Info("Resource not found. Creating a new one.")
		var observed T
		return SyncObservedResource(ctx, r, parentResource, observed, resource, opts...)
	}

	logs.Info(fmt.Sprintf("%s found.", resourceType))
	if toDelete {
		if !options.sharedOwnership {
			err = CheckControllerOwnership(oldResource, parentResource, r.Scheme())
			if err != nil {
				logs.Error(err, "Refusing to delete a resource with conflicting controller ownership")
				return
			}
		}
		logs.Info(fmt.Sprintf("%s found. Deleting the existing one.", resourceType))
		uid := oldResource.GetUID()
		resourceVersion := oldResource.GetResourceVersion()
		err = r.Delete(ctx, oldResource, client.Preconditions{UID: &uid, ResourceVersion: &resourceVersion})
		if err != nil {
			logs.Error(err, fmt.Sprintf("Failed to delete %s.", resourceType))
			r.GetRecorder().Eventf(oldResource, nil, corev1.EventTypeWarning, fmt.Sprintf("Delete%s", resourceType), "Delete", "Failed to delete %s %s: %s", resourceType, resourceNamespace, err)
			return
		}
		logs.Info(fmt.Sprintf("%s deleted.", resourceType))
		r.GetRecorder().Eventf(oldResource, nil, corev1.EventTypeNormal, fmt.Sprintf("Delete%s", resourceType), "Delete", "Deleted %s %s", resourceType, resourceNamespace)
		modified = true
		return
	}

	return SyncObservedResource(ctx, r, parentResource, oldResource, resource, opts...)
}

// SyncObservedResource synchronizes a desired resource against the exact
// object previously observed by its caller. Unlike SyncResource, it does not
// read from the API server. parentResource may be nil; when it is nil,
// SyncObservedResource neither sets nor validates controller ownership. Create
// and apply conflicts must be retried from a fresh observation so render-time
// decisions remain tied to the written object.
// desired must be non-nil rendered intent, without fields copied from a live
// object solely because the API server defaulted them. Explicit zero values
// must survive serialization through pointers or unstructured fields.
func SyncObservedResource[T client.Object](
	ctx context.Context,
	r Reconciler,
	parentResource client.Object,
	observed T,
	desired T,
	opts ...SyncOption,
) (bool, T, error) {
	// Rendered intent remains independent of the observed object and write responses.
	var zero T
	intent, ok := desired.DeepCopyObject().(T)
	if !ok {
		return false, zero, fmt.Errorf("deep copy desired resource as %T", desired)
	}
	gvk, err := apiutil.GVKForObject(desired, r.Scheme())
	if err != nil {
		return false, zero, err
	}
	options := resolveSyncOptions(opts)
	fieldManager := "dynamo-operator"
	if options.sharedOwnership && parentResource != nil {
		fieldManager += "-" + string(parentResource.GetUID())
	}

	// Create first so a stale cache miss cannot adopt a same-name foreign object.
	createdNew := isNilClientObject(observed)
	if createdNew {
		if parentResource != nil {
			if err := ctrl.SetControllerReference(parentResource, intent, r.Scheme()); err != nil {
				return false, zero, err
			}
		}
		hash, err := GetSpecHash(desired, opts...)
		if err != nil {
			return false, zero, err
		}
		updateAnnotations(intent, hash, 1)
		created := intent.DeepCopyObject().(T)
		if err := r.Create(ctx, created, client.FieldOwner(fieldManager)); err != nil {
			return false, zero, err
		}
		observed = created
	} else {
		// Validate ownership without turning observed lifecycle metadata into intent.
		if !options.sharedOwnership {
			if err := CheckControllerOwnership(observed, parentResource, r.Scheme()); err != nil {
				return false, zero, err
			}
		}
		if parentResource != nil && (!options.sharedOwnership || metav1.IsControlledBy(observed, parentResource)) {
			// Only our validated controller reference belongs to this actor's lifecycle intent.
			intent.SetOwnerReferences([]metav1.OwnerReference{*metav1.GetControllerOf(observed)})
		}

		// Preserve bookkeeping still read by existing callers, without gating metadata Apply.
		change, err := GetSpecChangeResult(observed, desired, opts...)
		if err != nil {
			return false, zero, err
		}
		if change.NewHash != nil {
			updateAnnotations(intent, *change.NewHash, change.NewGeneration)
		} else {
			annotations := intent.GetAnnotations()
			if annotations == nil {
				annotations = make(map[string]string, 2)
			}
			for _, key := range []string{NvidiaAnnotationHashKey, NvidiaAnnotationGenerationKey} {
				annotations[key] = observed.GetAnnotations()[key]
			}
			intent.SetAnnotations(annotations)
		}
	}

	// Transfer prior Update ownership so omitted fields can be removed by Apply.
	legacyOwner := !options.sharedOwnership || (parentResource != nil && metav1.IsControlledBy(observed, parentResource))
	observed, err = migrateUpdateOwnership(ctx, r, observed, fieldManager, legacyOwner)
	if err != nil {
		return false, zero, err
	}

	// Keep the exact observation's write preconditions while excluding server-owned fields.
	intent.GetObjectKind().SetGroupVersionKind(gvk)
	intent.SetUID(observed.GetUID())
	intent.SetResourceVersion(observed.GetResourceVersion())
	fields, err := runtime.DefaultUnstructuredConverter.ToUnstructured(intent)
	if err != nil {
		return false, zero, err
	}
	delete(fields, "status")
	metadata := fields["metadata"].(map[string]any)
	for _, key := range []string{"managedFields", "generation", "creationTimestamp", "deletionTimestamp", "deletionGracePeriodSeconds"} {
		delete(metadata, key)
	}
	applyObject := &unstructured.Unstructured{Object: fields}
	if err := r.Apply(ctx, client.ApplyConfigurationFromUnstructured(applyObject), client.FieldOwner(fieldManager), client.ForceOwnership); err != nil {
		return false, zero, err
	}

	// Continue from the write response, without reading it back through the cache.
	synced := desired.DeepCopyObject().(T)
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(applyObject.Object, synced); err != nil {
		return false, zero, err
	}
	comparison := synced.DeepCopyObject().(T)
	comparison.SetResourceVersion(observed.GetResourceVersion())
	comparison.SetManagedFields(observed.GetManagedFields())
	comparison.GetObjectKind().SetGroupVersionKind(observed.GetObjectKind().GroupVersionKind())
	modified := createdNew || (synced.GetResourceVersion() != observed.GetResourceVersion() && !equality.Semantic.DeepEqual(observed, comparison))
	if modified {
		action := "Update"
		if createdNew {
			action = "Create"
		}
		resourceType := gvk.Kind
		recordResourceEvent(r, synced, corev1.EventTypeNormal, action+resourceType, action, "%s %s %s/%s", action, resourceType, synced.GetNamespace(), synced.GetName())
	}
	return modified, synced, nil
}

// migrateUpdateOwnership upgrades this actor's Update entries and, for a legacy
// owner, the operator entries identified by its bookkeeping. Legacy lifecycle
// attribution is relinquished without deleting values: a shared Update manager
// does not tell us which controller added a finalizer or reference.
// observed must be non-nil.
func migrateUpdateOwnership[T client.Object](ctx context.Context, kubeClient client.Writer, observed T, fieldManager string, legacyOwner bool) (T, error) {
	// Shared resources must not transfer another parent's field ownership.
	managers := sets.New(fieldManager)
	if legacyOwner {
		bookkeeping := fieldpath.NewSet(
			fieldpath.MakePathOrDie("metadata", "annotations", NvidiaAnnotationHashKey),
			fieldpath.MakePathOrDie("metadata", "annotations", NvidiaAnnotationGenerationKey),
		)
		for _, owner := range csaupgrade.FindFieldsOwners(observed.GetManagedFields(), metav1.ManagedFieldsOperationUpdate, bookkeeping) {
			if owner.Subresource == "" {
				managers.Insert(owner.Manager)
			}
		}
	}

	// Do not transfer ambiguous lifecycle fields from a legacy shared manager to Apply.
	migrated := observed.DeepCopyObject().(T)
	lifecycle := fieldpath.NewSet(
		fieldpath.MakePathOrDie("metadata", "finalizers"),
		fieldpath.MakePathOrDie("metadata", "ownerReferences"),
	)
	entries := migrated.GetManagedFields()
	for i := range entries {
		entry := &entries[i]
		if entry.Operation != metav1.ManagedFieldsOperationUpdate || entry.Subresource != "" || !managers.Has(entry.Manager) {
			continue
		}
		fields := &fieldpath.Set{}
		if err := fields.FromJSON(entry.FieldsV1.GetRawReader()); err != nil {
			return observed, err
		}
		owned, err := fields.RecursiveDifference(lifecycle).ToJSON()
		if err != nil {
			return observed, err
		}
		entry.FieldsV1.SetRawBytes(owned)
	}
	migrated.SetManagedFields(entries)
	if err := csaupgrade.UpgradeManagedFields(migrated, managers, fieldManager); err != nil {
		return observed, err
	}
	if reflect.DeepEqual(observed.GetManagedFields(), migrated.GetManagedFields()) {
		return observed, nil
	}

	// An ownership-only patch preserves values and rejects a changed observation.
	if err := kubeClient.Patch(ctx, migrated, client.MergeFromWithOptions(observed, client.MergeFromWithOptimisticLock{})); err != nil {
		return observed, err
	}
	return migrated, nil
}

func isNilClientObject[T client.Object](object T) bool {
	value := reflect.ValueOf(object)
	return !value.IsValid() || ((value.Kind() == reflect.Chan ||
		value.Kind() == reflect.Func ||
		value.Kind() == reflect.Interface ||
		value.Kind() == reflect.Map ||
		value.Kind() == reflect.Pointer ||
		value.Kind() == reflect.Slice) && value.IsNil())
}

func recordResourceEvent(
	r Reconciler,
	object client.Object,
	eventType, reason, action, messageFmt string,
	args ...any,
) {
	if recorder := r.GetRecorder(); recorder != nil {
		recorder.Eventf(object, nil, eventType, reason, action, messageFmt, args...)
	}
}

// kubeEnvelopeFields are standard top-level Kubernetes fields that don't
// represent the resource's desired state. Everything else (spec, data,
// rules, roleRef, subjects, etc.) is considered content.
var kubeEnvelopeFields = map[string]bool{
	"apiVersion": true,
	"kind":       true,
	"metadata":   true,
	"status":     true,
}

// nonEnvelopeFields returns all top-level fields from an unstructured map
// except the Kubernetes envelope (apiVersion, kind, metadata, status).
func nonEnvelopeFields(obj map[string]any) map[string]any {
	content := make(map[string]any, len(obj))
	for k, v := range obj {
		if kubeEnvelopeFields[k] {
			continue
		}
		content[k] = v
	}
	return content
}

// getContentFields returns all content fields from an unstructured object,
// i.e. everything except the Kubernetes envelope (apiVersion, kind, metadata, status).
// For resources with a "spec" field, it returns the spec directly for
// backward-compatible hashing. For spec-less resources (ConfigMaps, Secrets,
// Roles, etc.), it returns a map of all content fields.
func getContentFields(u *unstructured.Unstructured) (any, bool) {
	if spec, found, err := unstructured.NestedFieldNoCopy(u.Object, "spec"); err == nil && found {
		return spec, true
	}

	content := nonEnvelopeFields(u.Object)
	if len(content) == 0 {
		return nil, false
	}
	return content, true
}

func getSpec(obj client.Object) (any, error) {
	sourceMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	sourceUnstructured := &unstructured.Unstructured{Object: sourceMap}

	content, found := getContentFields(sourceUnstructured)
	if !found {
		return nil, nil
	}
	return content, nil
}

// SpecChangeResult contains the result of spec change detection
type SpecChangeResult struct {
	// NewHash is the hash to set in the annotation (nil if no update needed)
	NewHash *string
	// NewGeneration is the generation to set in the annotation
	NewGeneration int64
	// NeedsUpdate indicates whether the resource needs to be updated
	NeedsUpdate bool
	// SpecNeedsUpdate indicates whether the desired spec/content must be copied.
	// When false with NeedsUpdate=true, only operator bookkeeping annotations need
	// repair and the resource spec should be left untouched.
	SpecNeedsUpdate bool
	// ManualChangeDetected indicates whether a manual change was detected
	ManualChangeDetected bool
}

// GetSpecChangeResult determines if a resource needs to be updated by comparing the desired spec hash
// with the last applied hash annotation. It also tracks generation to detect manual changes.
//
// Returns:
//   - SpecChangeResult with update information
//   - error if hash computation fails
func GetSpecChangeResult(current client.Object, desired client.Object, opts ...SyncOption) (SpecChangeResult, error) {
	desiredSpec, err := getSpec(desired)
	if err != nil {
		return SpecChangeResult{}, err
	}
	desiredHash, err := GetResourceHash(desiredSpec, opts...)
	if err != nil {
		return SpecChangeResult{}, err
	}
	currentSpec, err := getSpec(current)
	if err != nil {
		return SpecChangeResult{}, err
	}
	currentMatchesDesired := equality.Semantic.DeepEqual(currentSpec, desiredSpec)

	lastAppliedHash := getAnnotation(current, NvidiaAnnotationHashKey)
	lastAppliedGenStr := getAnnotation(current, NvidiaAnnotationGenerationKey)
	currentGen := current.GetGeneration()
	annotationOnlyChange := func() SpecChangeResult {
		return SpecChangeResult{
			NewHash:       &desiredHash,
			NewGeneration: currentGen,
			NeedsUpdate:   true,
		}
	}
	specChange := func(manual bool) SpecChangeResult {
		return SpecChangeResult{
			NewHash:              &desiredHash,
			NewGeneration:        currentGen + 1,
			NeedsUpdate:          true,
			SpecNeedsUpdate:      true,
			ManualChangeDetected: manual,
		}
	}

	// Case 1: Hash annotation missing (external create or pre-upgrade resource)
	// Note: This is not first-time CREATE (handled separately in SyncResource with generation=1).
	// If the live spec already matches the desired spec, only backfill the operator
	// bookkeeping annotations. This avoids rewriting API-server-defaulted fields
	// during upgrades.
	if lastAppliedHash == "" {
		if currentMatchesDesired {
			return annotationOnlyChange(), nil
		}
		return specChange(false), nil
	}

	// Case 2: Hash different (spec changed)
	if desiredHash != lastAppliedHash {
		if currentMatchesDesired {
			return annotationOnlyChange(), nil
		}
		return specChange(false), nil
	}

	// Case 3: Hash same, but generation annotation missing (upgrade scenario)
	if lastAppliedGenStr == "" {
		if currentMatchesDesired {
			return annotationOnlyChange(), nil
		}
		return specChange(false), nil
	}

	// Case 4: Both annotations exist, check for manual changes
	lastAppliedGen, err := strconv.ParseInt(lastAppliedGenStr, 10, 64)
	if err != nil {
		// Corrupted annotation, force update to fix
		if currentMatchesDesired {
			return annotationOnlyChange(), nil
		}
		return specChange(false), nil
	}

	// Detect manual changes: if current generation > last applied generation,
	// someone else modified the resource after our last update. Some resources
	// can advance generation for metadata-only writes, so confirm that the
	// current content actually drifted before forcing a rewrite.
	if currentGen > 0 && currentGen > lastAppliedGen {
		if currentMatchesDesired {
			return annotationOnlyChange(), nil
		}
		return specChange(true), nil
	}

	// No update needed
	return SpecChangeResult{
		NeedsUpdate: false,
	}, nil
}

// getAnnotation safely retrieves an annotation value from an object
func getAnnotation(obj client.Object, key string) string {
	annotations := obj.GetAnnotations()
	if annotations == nil {
		return ""
	}
	return annotations[key]
}

func GetSpecHash(obj client.Object, opts ...SyncOption) (string, error) {
	spec, err := getSpec(obj)
	if err != nil {
		return "", err
	}
	return GetResourceHash(spec, opts...)
}

// updateAnnotations sets both hash and generation annotations on an object
func updateAnnotations(obj client.Object, hash string, generation int64) {
	annotations := obj.GetAnnotations()
	if annotations == nil {
		annotations = map[string]string{}
	}
	annotations[NvidiaAnnotationHashKey] = hash
	annotations[NvidiaAnnotationGenerationKey] = strconv.FormatInt(generation, 10)
	obj.SetAnnotations(annotations)
}

// GetResourceHash returns a consistent hash for the given object spec
func GetResourceHash(obj any, opts ...SyncOption) (string, error) {
	// Convert obj to a map[string]interface{}
	objMap, err := json.Marshal(obj)
	if err != nil {
		return "", err
	}

	// JSON already orders map keys deterministically without reordering lists.
	if resolveSyncOptions(opts).preserveListOrder {
		return fmt.Sprintf("%x", sha256.Sum256(objMap)), nil
	}

	var objData map[string]any
	if err := json.Unmarshal(objMap, &objData); err != nil {
		return "", err
	}

	// Sort keys to ensure consistent serialization
	sortedObjData := SortKeys(objData)

	// Serialize to JSON
	serialized, err := json.Marshal(sortedObjData)
	if err != nil {
		return "", err
	}

	// Compute the hash
	hasher := sha256.New()
	hasher.Write(serialized)
	return fmt.Sprintf("%x", hasher.Sum(nil)), nil
}

// SortKeys consumes a JSON-decoded tree, ordering slices for consistent serialization.
func SortKeys(obj any) any {
	switch obj := obj.(type) {
	case map[string]any:
		if obj == nil {
			return map[string]any{}
		}
		for k, v := range obj {
			obj[k] = SortKeys(v)
		}
		return obj
	case []any:
		// Check if the slice contains maps and sort them by the "name" field or the first available field
		if len(obj) > 0 {

			if _, ok := obj[0].(map[string]any); ok {
				sort.SliceStable(obj, func(i, j int) bool {
					iMap, iOk := obj[i].(map[string]any)
					jMap, jOk := obj[j].(map[string]any)
					if iOk && jOk {
						// Try to sort by "name" if present
						iName, iNameOk := iMap["name"].(string)
						jName, jNameOk := jMap["name"].(string)
						if iNameOk && jNameOk {
							return iName < jName
						}

						// If "name" is not available, sort by the first key in each map
						if len(iMap) > 0 && len(jMap) > 0 {
							iFirstKey := firstKey(iMap)
							jFirstKey := firstKey(jMap)
							return iFirstKey < jFirstKey
						}
					}
					// If no valid comparison is possible, maintain the original order
					return false
				})
			}
		}
		for i, v := range obj {
			obj[i] = SortKeys(v)
		}
	}
	return obj
}

// Helper function to get the first key of a map (alphabetically sorted)
func firstKey(m map[string]any) string {
	var first string
	firstSet := false
	for k := range m {
		if !firstSet || k < first {
			first, firstSet = k, true
		}
	}
	return first
}

// AppendUniqueImagePullSecrets appends secrets to existing, skipping any that already exist by name.
func AppendUniqueImagePullSecrets(existing, additional []corev1.LocalObjectReference) []corev1.LocalObjectReference {
	if len(additional) == 0 {
		return existing
	}
	seen := make(map[string]bool, len(existing))
	for _, s := range existing {
		seen[s.Name] = true
	}
	for _, s := range additional {
		if !seen[s.Name] {
			existing = append(existing, s)
			seen[s.Name] = true
		}
	}
	return existing
}

type Resource struct {
	object            client.Object
	isReady           bool
	readyReason       string
	componentStatuses map[string]v1beta1.ComponentReplicaStatus
}

func NewResource[T client.Object](resource T, isReady func() (bool, string)) (*Resource, error) {
	v := reflect.ValueOf(resource)
	// handles untype nil and typed nil
	if !v.IsValid() || v.IsNil() {
		return nil, fmt.Errorf("resource is nil")
	}
	ready, reason := isReady()
	return &Resource{
		object:      resource,
		isReady:     ready,
		readyReason: reason,
	}, nil
}

func NewResourceWithComponentStatuses[T client.Object](resource T, isReadyAndComponentStatuses func() (bool, string, map[string]v1beta1.ComponentReplicaStatus)) (*Resource, error) {
	v := reflect.ValueOf(resource)
	// handles untype nil and typed nil
	if !v.IsValid() || v.IsNil() {
		return nil, fmt.Errorf("resource is nil")
	}
	ready, reason, componentStatuses := isReadyAndComponentStatuses()
	return &Resource{
		object:            resource,
		isReady:           ready,
		readyReason:       reason,
		componentStatuses: componentStatuses,
	}, nil
}

func (r *Resource) IsReady() (bool, string) {
	return r.isReady, r.readyReason
}

func (r *Resource) GetName() string {
	return r.object.GetName()
}

func (r *Resource) GetComponentStatuses() map[string]v1beta1.ComponentReplicaStatus {
	return r.componentStatuses
}
