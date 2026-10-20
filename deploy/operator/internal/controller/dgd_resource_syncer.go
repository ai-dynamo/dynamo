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

package controller

import (
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// dgdResourceSyncer is the common write capability required by concrete DGD
// resource reconcilers that use controller_common.SyncResource. It carries no
// DGD orchestration or provider behavior.
type dgdResourceSyncer struct {
	client.Client
	recorder events.EventRecorder
}

func newDGDResourceSyncer(kubeClient client.Client, recorder events.EventRecorder) dgdResourceSyncer {
	return dgdResourceSyncer{Client: kubeClient, recorder: recorder}
}

func (s *dgdResourceSyncer) GetRecorder() events.EventRecorder {
	return s.recorder
}

// deleteIfControlledBy deletes the object at key, but only the exact object it
// verified owner controls. It reads into obj, so obj must be a non-nil, empty
// object of the target type; owner must be non-nil. kind labels error messages.
//
// The UID precondition is what makes the check and the delete one decision: if
// the object is replaced by an unowned object of the same name in between, the
// delete fails with a conflict and the reconcile retries against the
// replacement instead of removing an object owner does not control. An absent
// object, or one owner does not control, is success, not an error.
func (s *dgdResourceSyncer) deleteIfControlledBy(
	ctx context.Context,
	owner metav1.Object,
	key types.NamespacedName,
	obj client.Object,
	kind string,
) error {
	err := s.Get(ctx, key, obj)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to get %s %s: %w", kind, key.Name, err)
	}
	if !metav1.IsControlledBy(obj, owner) {
		return nil
	}

	uid := obj.GetUID()
	err = s.Delete(ctx, obj, client.Preconditions{UID: &uid})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("failed to delete %s %s: %w", kind, key.Name, err)
	}
	return nil
}
