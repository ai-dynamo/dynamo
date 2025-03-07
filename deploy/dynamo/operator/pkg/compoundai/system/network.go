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
	"context"

	corev1 "k8s.io/api/core/v1"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/operator/pkg/dynamo/consts"
)

func GetNetworkConfigConfigMap(ctx context.Context, configmapGetter func(ctx context.Context, namespace, name string) (*corev1.ConfigMap, error)) (configMap *corev1.ConfigMap, err error) {
	configMap, err = configmapGetter(ctx, GetNamespace(), consts.KubeConfigMapNameNetworkConfig)
	return
}
