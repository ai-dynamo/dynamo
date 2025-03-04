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

package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"

type SubscriptionAction string

const (
	SubscriptionActionSubscribe   SubscriptionAction = "subscribe"
	SubscriptionActionUnsubscribe SubscriptionAction = "unsubscribe"
)

type SubscriptionRespSchema struct {
	ResourceType modelschemas.ResourceType `json:"resource_type"`
	Payload      interface{}               `json:"payload"`
}

type SubscriptionReqSchema struct {
	WsReqSchema
	Payload *struct {
		Action       SubscriptionAction        `json:"action"`
		ResourceType modelschemas.ResourceType `json:"resource_type"`
		ResourceUids []string                  `json:"resource_uids"`
	} `json:"payload"`
}
