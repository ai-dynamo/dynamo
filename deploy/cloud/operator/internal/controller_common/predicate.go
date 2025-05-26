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
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

type Config struct {
	// Enable resources filtering, only the resources belonging to the given namespace will be handled.
	RestrictedNamespace string
	// If true, assume VirtualService endpoints are HTTPS
	VirtualServiceSupportsHTTPS bool
	EnableLWS                   bool
}

func EphemeralDeploymentEventFilter(config Config) predicate.Predicate {
	return predicate.NewPredicateFuncs(func(o client.Object) bool {
		l := log.FromContext(context.Background())
		objMeta, err := meta.Accessor(o)
		if err != nil {
			l.Error(err, "Error extracting object metadata")
			return false
		}
		if config.RestrictedNamespace != "" {
			// in case of a restricted namespace, we only want to process the events that are in the restricted namespace
			return objMeta.GetNamespace() == config.RestrictedNamespace
		}
		// in all other cases, discard the event if it is destined to an ephemeral deployment
		if strings.Contains(objMeta.GetNamespace(), "ephemeral") {
			return false
		}
		return true
	})
}
