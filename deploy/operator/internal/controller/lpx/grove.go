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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
)

// reconcileGrovePodCliqueSetForLPX converges the already-rendered PodCliqueSet
// through native Grove rollout. existingPodCliqueSet may be nil when absent;
// nil replicas leaves scale to Grove. deployment and desired are non-nil.
func (r *graphReconciler) reconcileGrovePodCliqueSetForLPX(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	replicas *int32,
	existingPodCliqueSet *grovev1alpha1.PodCliqueSet,
	desired *grovev1alpha1.PodCliqueSet,
) (*grovev1alpha1.PodCliqueSet, bool, *lpxRetiring, error) {
	// A missing observation permits only creation after the publication fence.
	if existingPodCliqueSet == nil {
		if err := r.validateLPXPublicationSource(ctx, deployment); err != nil {
			return nil, false, nil, err
		}
		retiring, err := r.retireFirstOwnedLPXRequest(ctx, deployment, desired.Name, "LPX publication was retired before synchronizing a Grove PodCliqueSet spec change")
		if err != nil || retiring != nil {
			return nil, false, retiring, err
		}
	}

	// Grove seeds native scale only at creation; preserve that seed when replicas are omitted.
	if existingPodCliqueSet != nil && replicas == nil {
		group := &desired.Spec.Template.PodCliqueScalingGroupConfigs[0]
		for _, existing := range existingPodCliqueSet.Spec.Template.PodCliqueScalingGroupConfigs {
			if existing.Name == group.Name {
				group.Replicas = existing.Replicas
				break
			}
		}
	}

	// Match normal PCS synchronization; Grove owns rollout and immutable-field validation.
	modified, synced, err := commoncontroller.SyncObservedResource(ctx, r, deployment, existingPodCliqueSet, desired, commoncontroller.WithPreservedListOrder())
	return synced, modified, nil, err
}
