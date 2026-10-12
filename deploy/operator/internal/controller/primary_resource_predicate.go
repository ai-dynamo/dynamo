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
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// DGD renderers consume metadata as well as spec; status alone is not input.
func dgdPrimaryPredicate() predicate.Predicate {
	return predicate.Or(
		commoncontroller.GenerationOrDeletionChangedPredicate(),
		predicate.AnnotationChangedPredicate{},
		predicate.LabelChangedPredicate{},
	)
}
