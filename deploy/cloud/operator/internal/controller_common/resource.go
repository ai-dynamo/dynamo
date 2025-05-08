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
	"fmt"
	"reflect"

	"github.com/cisco-open/k8s-objectmatcher/patch"
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

var (
	annotator  = patch.NewAnnotator(NvidiaAnnotationHashKey)
	patchMaker = patch.NewPatchMaker(annotator, &patch.K8sStrategicMergePatcher{}, &patch.BaseJSONMergePatcher{})
)

// IsSpecChanged returns true if the spec has changed between the existing one
// and the new resource spec compared by hash.
func IsSpecChanged(current client.Object, desired client.Object) (bool, error) {
	if current == nil && desired != nil {
		return true, nil
	}

	var patchResult *patch.PatchResult
	opts := []patch.CalculateOption{
		patch.IgnoreStatusFields(),
		patch.IgnoreField("metadata"),
	}
	patchResult, err := patchMaker.Calculate(current, desired, opts...)
	if err != nil {
		return false, fmt.Errorf("failed to calculate patch: %w", err)
	}
	logs := log.FromContext(context.Background())
	logs.Info("patchResult", "patchResult", patchResult)
	return !patchResult.IsEmpty(), nil
}

type Reconciler interface {
	client.Client
	GetRecorder() record.EventRecorder
}

//nolint:nakedret
func SyncResource[T client.Object](ctx context.Context, r Reconciler, parentResource client.Object, generateResource func(ctx context.Context) (T, bool, error)) (modified bool, res T, err error) {
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

		err = annotator.SetLastAppliedAnnotation(resource)
		if err != nil {
			logs.Error(err, "Failed to set last applied annotation.")
			r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for %s %s: %s", resourceType, resourceNamespace, err)
			return
		}

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
		var changed bool
		changed, err = IsSpecChanged(oldResource, resource)
		if err != nil {
			r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, fmt.Sprintf("CalculatePatch%s", resourceType), "Failed to calculate patch for %s %s: %s", resourceType, resourceNamespace, err)
			return false, resource, fmt.Errorf("failed to check if spec has changed: %w", err)
		}
		if changed {
			// update the spec of the current object with the desired spec
			err = CopySpec(resource, oldResource)
			if err != nil {
				logs.Error(err, fmt.Sprintf("Failed to copy spec for %s.", resourceType))
				r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, fmt.Sprintf("CopySpec%s", resourceType), "Failed to copy spec for %s %s: %s", resourceType, resourceNamespace, err)
				return
			}
			err = annotator.SetLastAppliedAnnotation(oldResource)
			if err != nil {
				logs.Error(err, "Failed to set last applied annotation.")
				r.GetRecorder().Eventf(parentResource, corev1.EventTypeWarning, "SetLastAppliedAnnotation", "Failed to set last applied annotation for %s %s: %s", resourceType, resourceNamespace, err)
				return
			}
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
