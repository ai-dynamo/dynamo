/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
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
 * Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
 */

package api_store_client

import (
	"context"
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/schemas"
)

type ApiStoreClient struct {
	endpoint string
}

func NewApiStoreClient(endpoint string) *ApiStoreClient {
	return &ApiStoreClient{
		endpoint: endpoint,
	}
}

func (c *ApiStoreClient) GetDynamoComponent(ctx context.Context, name, version string) (component *schemas.DynamoComponent, err error) {
	url_ := urlJoin(c.endpoint, fmt.Sprintf("/api/v1/dynamo_nims/%s/versions/%s", name, version))
	component = &schemas.DynamoComponent{}
	_, err = DoJsonRequest(ctx, "GET", url_, nil, nil, nil, component, nil)
	return
}

func (c *ApiStoreClient) PresignDynamoComponentDownloadURL(ctx context.Context, name, version string) (component *schemas.DynamoComponent, err error) {
	url_ := urlJoin(c.endpoint, fmt.Sprintf("/api/v1/dynamo_nims/%s/versions/%s/presign_download_url", name, version))
	component = &schemas.DynamoComponent{}
	_, err = DoJsonRequest(ctx, "PATCH", url_, nil, nil, nil, component, nil)
	return
}

func urlJoin(baseURL string, pathPart string) string {
	return strings.TrimRight(baseURL, "/") + "/" + strings.TrimLeft(pathPart, "/")
}
