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
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func TestGenerationOrDeletionChangedPredicate(t *testing.T) {
	pred := generationOrDeletionChangedPredicate()
	oldObject := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Generation: 1}}

	t.Run("preserves non-update event handling", func(t *testing.T) {
		assert.True(t, pred.Create(event.CreateEvent{Object: oldObject}))
		assert.True(t, pred.Delete(event.DeleteEvent{Object: oldObject}))
		assert.True(t, pred.Generic(event.GenericEvent{Object: oldObject}))
	})

	t.Run("rejects same-generation updates", func(t *testing.T) {
		assert.False(t, pred.Update(event.UpdateEvent{
			ObjectOld: oldObject,
			ObjectNew: oldObject.DeepCopy(),
		}))
	})

	t.Run("admits generation changes", func(t *testing.T) {
		newObject := oldObject.DeepCopy()
		newObject.Generation++
		assert.True(t, pred.Update(event.UpdateEvent{
			ObjectOld: oldObject,
			ObjectNew: newObject,
		}))
	})

	t.Run("admits deletion start without a generation change", func(t *testing.T) {
		newObject := oldObject.DeepCopy()
		now := metav1.Now()
		newObject.DeletionTimestamp = &now
		assert.True(t, pred.Update(event.UpdateEvent{
			ObjectOld: oldObject,
			ObjectNew: newObject,
		}))
	})

	t.Run("rejects later same-generation deletion updates", func(t *testing.T) {
		deletingObject := oldObject.DeepCopy()
		now := metav1.Now()
		deletingObject.DeletionTimestamp = &now
		assert.False(t, pred.Update(event.UpdateEvent{
			ObjectOld: deletingObject,
			ObjectNew: deletingObject.DeepCopy(),
		}))
	})
}
