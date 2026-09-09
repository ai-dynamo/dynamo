/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package lpx

import (
	"context"
	"fmt"
	"strconv"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// reconcileGrovePodCliqueSetForLPX converges the already-rendered PodCliqueSet
// behind the LPX retirement fence. Only existingPodCliqueSet may be nil, when
// the controller observed no PCS; all decisions and writes use that observation.
func (r *graphReconciler) reconcileGrovePodCliqueSetForLPX(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	existingPodCliqueSet *grovev1alpha1.PodCliqueSet,
	desired *grovev1alpha1.PodCliqueSet,
	selectedLPX *lpxMaterializing,
) (*grovev1alpha1.PodCliqueSet, bool, error) {
	// A missing observation permits only creation after the publication fence.
	if existingPodCliqueSet == nil {
		if err := r.fenceLPXPublicationBeforeGroveSpecWrite(ctx, deployment); err != nil {
			return nil, false, err
		}
		modified, synced, err := commoncontroller.SyncObservedResource(ctx, r, deployment, existingPodCliqueSet, desired)
		return synced, modified, err
	}

	// Retire and replace a changed attempt or spec; never update its spec in place.
	change, err := commoncontroller.GetSpecChangeResult(existingPodCliqueSet, desired)
	if err != nil {
		return nil, false, fmt.Errorf("compare cached LPX PodCliqueSet with its successor: %w", err)
	}
	if change.SpecNeedsUpdate || !selectedLPX.hasCurrentLPXAttemptAnnotations(existingPodCliqueSet, deployment) {
		return nil, false, r.recreateStaleLPXPodCliqueSet(ctx, deployment, desired, selectedLPX)
	}

	// A current PCS keeps its spec and identity; only owned metadata may change.
	if err := commoncontroller.CheckControllerOwnership(existingPodCliqueSet, deployment, r.Scheme()); err != nil {
		return nil, false, err
	}
	prepared := existingPodCliqueSet
	if change.NeedsUpdate {
		prepared = existingPodCliqueSet.DeepCopy()
		metav1.SetMetaDataAnnotation(&prepared.ObjectMeta, commoncontroller.NvidiaAnnotationHashKey, *change.NewHash)
		metav1.SetMetaDataAnnotation(&prepared.ObjectMeta, commoncontroller.NvidiaAnnotationGenerationKey, strconv.FormatInt(change.NewGeneration, 10))
	}
	for _, key := range [...]string{
		dynamolpx.WorkloadDigestAnnotation,
		lpxDeploymentUIDAnnotation,
		lpxDeploymentGenerationAnnotation,
		dynamo.LPXInputRevisionAnnotation,
		dynamolpx.DGDUIDAnnotation,
		dynamolpx.DGDGenerationAnnotation,
	} {
		value, found := desired.Annotations[key]
		currentValue, currentFound := prepared.Annotations[key]
		if found == currentFound && (!found || value == currentValue) {
			continue
		}
		if prepared == existingPodCliqueSet {
			prepared = existingPodCliqueSet.DeepCopy()
		}
		if found {
			metav1.SetMetaDataAnnotation(&prepared.ObjectMeta, key, value)
		} else {
			delete(prepared.Annotations, key)
		}
	}
	if prepared == existingPodCliqueSet {
		return existingPodCliqueSet, false, nil
	}

	// The observed resource version prevents overwriting any concurrent PCS edit.
	if err := r.Update(ctx, prepared); err != nil {
		r.GetRecorder().Eventf(existingPodCliqueSet, nil, corev1.EventTypeWarning, "UpdatePodCliqueSet", "Update", "Failed to update metadata for PodCliqueSet %s: %s", desired.Namespace, err)
		return nil, false, err
	}
	r.GetRecorder().Eventf(prepared, nil, corev1.EventTypeNormal, "UpdatePodCliqueSet", "Update", "Updated metadata for PodCliqueSet %s", desired.Namespace)
	return prepared, true, nil
}
