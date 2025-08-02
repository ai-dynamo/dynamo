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
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/client-go/discovery"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

type GroveConfig struct {
	// Enabled is automatically determined by checking if Grove CRDs are installed in the cluster
	Enabled bool
	// TerminationDelay configures the termination delay for Grove PodGangSets
	TerminationDelay time.Duration
}

type Config struct {
	// Enable resources filtering, only the resources belonging to the given namespace will be handled.
	RestrictedNamespace string
	EnableLWS           bool
	Grove               GroveConfig
	EtcdAddress         string
	NatsAddress         string
	IngressConfig       IngressConfig
}

type IngressConfig struct {
	VirtualServiceGateway      string
	IngressControllerClassName string
	IngressControllerTLSSecret string
	IngressHostSuffix          string
}

func (i *IngressConfig) UseVirtualService() bool {
	return i.VirtualServiceGateway != ""
}

// DetectGroveAvailability checks if Grove is available by checking if the Grove API group is registered
// This approach uses the discovery client which is simpler and more reliable
func DetectGroveAvailability(ctx context.Context, mgr ctrl.Manager) bool {
	logger := log.FromContext(ctx)

	// Use the discovery client to check if Grove API groups are available
	cfg := mgr.GetConfig()
	if cfg == nil {
		logger.Info("Grove detection failed, no discovery client available")
		return false
	}

	// Try to create a discovery client
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(cfg)
	if err != nil {
		logger.Error(err, "Grove detection failed, could not create discovery client")
		return false
	}

	// Check if grove.io API group is available
	apiGroups, err := discoveryClient.ServerGroups()
	if err != nil {
		logger.Error(err, "Grove detection failed, could not list server groups")
		return false
	}

	for _, group := range apiGroups.Groups {
		if group.Name == "grove.io" {
			logger.Info("Grove is available, grove.io API group found")
			return true
		}
	}

	logger.Info("Grove not available, grove.io API group not found")
	return false
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
