/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// NvidiaAnnotationHashKey indicates annotation name for last applied hash by the operator
	NvidiaAnnotationHashKey = "nvidia.com/last-applied-hash"
)

type Reconciler interface {
	client.Client
	GetRecorder() record.EventRecorder
}

// ResourceGenerator is a function that generates a resource.
// it must return the resource, a boolean indicating if the resource should be deleted, and an error
// if the resource should be deleted, the returned resource must contain the necessary information to delete it (name and namespace)
type ResourceGenerator[T client.Object] func(ctx context.Context) (T, bool, error)

//nolint:nakedret
func SyncResource[T client.Object](ctx context.Context, r Reconciler, parentResource client.Object, generateResource ResourceGenerator[T]) (modified bool, res T, err error) {
	logs := log.FromContext(ctx)

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
		r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, fmt.Sprintf("Get%s", resourceType), "Failed to get %s %s: %s", resourceType, resourceNamespace, err)
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

		err = ctrl.SetControllerReference(parentResource, resource, r.Scheme())
		if err != nil {
			logs.Error(err, "Failed to set controller reference.")
			r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, "SetControllerReference", "Failed to set controller reference for %s %s: %s", resourceType, resourceNamespace, err)
			return
		}

		var hash string
		hash, err = GetSpecHash(resource)
		if err != nil {
			logs.Error(err, "Failed to get spec hash.")
			r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, "GetSpecHash", "Failed to get spec hash for %s %s: %s", resourceType, resourceNamespace, err)
			return
		}

		updateHashAnnotation(resource, hash)

		r.GetRecorder().Eventf(parentResource, corev1.EventTypeNormal, fmt.Sprintf("Create%s", resourceType), "Creating a new %s %s", resourceType, resourceNamespace)
		err = r.Create(ctx, resource)
		if err != nil {
			logs.Error(err, "Failed to create Resource.")
			r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, fmt.Sprintf("Create%s", resourceType), "Failed to create %s %s: %s", resourceType, resourceNamespace, err)
			return
		}
		logs.Info(fmt.Sprintf("%s created.", resourceType))
		r.GetRecorder().Eventf(parentResource, corev1.EventTypeNormal, fmt.Sprintf("Create%s", resourceType), "Created %s %s", resourceType, resourceNamespace)
		modified = true
		res = resource
	} else {
		logs.Info(fmt.Sprintf("%s found.", resourceType))
		if toDelete {
			logs.Info(fmt.Sprintf("%s not found. Deleting the existing one.", resourceType))
			err = r.Delete(ctx, oldResource)
			if err != nil {
				logs.Error(err, fmt.Sprintf("Failed to delete %s.", resourceType))
				r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, fmt.Sprintf("Delete%s", resourceType), "Failed to delete %s %s: %s", resourceType, resourceNamespace, err)
				return
			}
			logs.Info(fmt.Sprintf("%s deleted.", resourceType))
			r.GetRecorder().Eventf(parentResource, corev1.EventTypeNormal, fmt.Sprintf("Delete%s", resourceType), "Deleted %s %s", resourceType, resourceNamespace)
			modified = true
			return
		}

		// Check if the Spec has changed and update if necessary
		var newHash *string
		newHash, err = IsSpecChanged(oldResource, resource)
		if err != nil {
			r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, fmt.Sprintf("CalculatePatch%s", resourceType), "Failed to calculate patch for %s %s: %s", resourceType, resourceNamespace, err)
			return false, resource, fmt.Errorf("failed to check if spec has changed: %w", err)
		}
		if newHash != nil {
			// update the spec of the current object with the desired spec
			err = CopySpec(resource, oldResource)
			if err != nil {
				logs.Error(err, fmt.Sprintf("Failed to copy spec for %s.", resourceType))
				r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, fmt.Sprintf("CopySpec%s", resourceType), "Failed to copy spec for %s %s: %s", resourceType, resourceNamespace, err)
				return
			}

			updateHashAnnotation(oldResource, *newHash)

			err = r.Update(ctx, oldResource)
			if err != nil {
				logs.Error(err, fmt.Sprintf("Failed to update %s.", resourceType))
				r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, fmt.Sprintf("Update%s", resourceType), "Failed to update %s %s: %s", resourceType, resourceNamespace, err)
				return
			}
			logs.Info(fmt.Sprintf("%s updated.", resourceType))
			r.GetRecorder().Eventf(parentResource, corev1.EventTypeNormal, fmt.Sprintf("Update%s", resourceType), "Updated %s %s", resourceType, resourceNamespace)
			modified = true
			res = oldResource
		} else {
			logs.Info(fmt.Sprintf("%s spec is the same. Skipping update.", resourceType))
			r.GetRecorder().Eventf(parentResource, corev1.EventTypeNormal, fmt.Sprintf("Update%s", resourceType), "Skipping update %s %s", resourceType, resourceNamespace)
			res = oldResource
		}
	}
	return
}

// CopySpec copies only the Spec field from source to destination using Unstructured
func CopySpec(source, destination client.Object) error {
	// Convert source to unstructured
	sourceMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(source)
	if err != nil {
		return err
	}
	sourceUnstructured := &unstructured.Unstructured{Object: sourceMap}

	// Convert destination to unstructured
	destMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(destination)
	if err != nil {
		return err
	}
	destUnstructured := &unstructured.Unstructured{Object: destMap}

	// Extract only the spec from source
	sourceSpec, found, err := unstructured.NestedFieldCopy(sourceUnstructured.Object, "spec")
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("spec not found in source object")
	}

	// Set the spec in the destination
	err = unstructured.SetNestedField(destUnstructured.Object, sourceSpec, "spec")
	if err != nil {
		return err
	}

	// Convert back to the original object
	return runtime.DefaultUnstructuredConverter.FromUnstructured(destUnstructured.Object, destination)
}

func getSpec(obj client.Object) (any, error) {
	// Convert source to unstructured
	sourceMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	sourceUnstructured := &unstructured.Unstructured{Object: sourceMap}
	// Extract only the spec from source
	spec, found, err := unstructured.NestedFieldCopy(sourceUnstructured.Object, "spec")
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, nil
	}
	return spec, nil
}

// IsSpecChanged returns the new hash if the spec has changed between the existing one
func IsSpecChanged(current client.Object, desired client.Object) (*string, error) {
	hashStr, err := GetSpecHash(desired)
	if err != nil {
		return nil, err
	}
	if currentHash, ok := current.GetAnnotations()[NvidiaAnnotationHashKey]; ok {
		if currentHash == hashStr {
			return nil, nil
		}
	}
	return &hashStr, nil
}

func GetSpecHash(obj client.Object) (string, error) {
	spec, err := getSpec(obj)
	if err != nil {
		return "", err
	}
	return GetResourceHash(spec)
}

func updateHashAnnotation(obj client.Object, hash string) {
	annotations := obj.GetAnnotations()
	if annotations == nil {
		annotations = map[string]string{}
	}
	annotations[NvidiaAnnotationHashKey] = hash
	obj.SetAnnotations(annotations)
}

// GetResourceHash returns a consistent hash for the given object spec
func GetResourceHash(obj any) (string, error) {
	// Convert obj to a map[string]interface{}
	objMap, err := json.Marshal(obj)
	if err != nil {
		return "", err
	}

	var objData map[string]interface{}
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

// SortKeys recursively sorts the keys of a map to ensure consistent serialization
func SortKeys(obj interface{}) interface{} {
	switch obj := obj.(type) {
	case map[string]interface{}:
		sortedMap := make(map[string]interface{})
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			sortedMap[k] = SortKeys(obj[k])
		}
		return sortedMap
	case []interface{}:
		// Check if the slice contains maps and sort them by the "name" field or the first available field
		if len(obj) > 0 {

			if _, ok := obj[0].(map[string]interface{}); ok {
				sort.SliceStable(obj, func(i, j int) bool {
					iMap, iOk := obj[i].(map[string]interface{})
					jMap, jOk := obj[j].(map[string]interface{})
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
func firstKey(m map[string]interface{}) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys[0]
}
