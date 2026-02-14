/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package v1beta1

// Hub marks DynamoCheckpoint as the hub type for conversion.
func (*DynamoCheckpoint) Hub() {}

// Hub marks DynamoComponentDeployment as the hub type for conversion.
func (*DynamoComponentDeployment) Hub() {}

// Hub marks DynamoGraphDeployment as the hub type for conversion.
func (*DynamoGraphDeployment) Hub() {}

// Hub marks DynamoGraphDeploymentScalingAdapter as the hub type for conversion.
func (*DynamoGraphDeploymentScalingAdapter) Hub() {}

// Hub marks DynamoGraphDeploymentRequest as the hub type for conversion.
func (*DynamoGraphDeploymentRequest) Hub() {}

// Hub marks DynamoModel as the hub type for conversion.
func (*DynamoModel) Hub() {}
