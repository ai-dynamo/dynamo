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

package system

import (
	"os"
	"sync"

	"github.com/sirupsen/logrus"
)

const (
	// NamespaceEnvKey is the environment variable that specifies the system namespace.
	NamespaceEnvKey = "SYSTEM_NAMESPACE"
	// ResourceLabelEnvKey is the environment variable that specifies the system resource
	// label.
	ResourceLabelEnvKey = "SYSTEM_RESOURCE_LABEL"

	DefaultNamespace = "yatai-deployment"
	MagicDNSEnvKey   = "MAGIC_DNS"
	DefaultMagicDNS  = "sslip.io"
)

var (
	once sync.Once
)

// GetNamespace returns the name of the K8s namespace where our system components
// run.
func GetNamespace() string {
	if ns := os.Getenv(NamespaceEnvKey); ns != "" {
		return ns
	}

	once.Do(func() {
		logrus.Infof("%s environment variable not set, using default namespace %s", NamespaceEnvKey, DefaultNamespace)
	})
	return DefaultNamespace
}

// GetResourceLabel returns the label key identifying K8s objects our system
// components source their configuration from.
func GetResourceLabel() string {
	return os.Getenv(ResourceLabelEnvKey)
}

func GetMagicDNS() string {
	magicDNS := os.Getenv(MagicDNSEnvKey)
	if magicDNS == "" {
		magicDNS = DefaultMagicDNS
	}
	return magicDNS
}
