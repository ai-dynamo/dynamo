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
