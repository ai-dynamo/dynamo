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

type CompletePartSchema struct {
	PartNumber int    `json:"part_number" binding:"required"`
	ETag       string `json:"etag" binding:"required"`
}

type PreSignMultipartUploadSchema struct {
	UploadId   string `json:"upload_id" binding:"required"`
	PartNumber int    `json:"part_number" binding:"required"`
}

type CompleteMultipartUploadSchema struct {
	UploadId string               `json:"upload_id" binding:"required"`
	Parts    []CompletePartSchema `json:"parts" binding:"required"`
}
