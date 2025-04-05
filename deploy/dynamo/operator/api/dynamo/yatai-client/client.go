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

package yataiclient

import (
	"context"
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/schemas"
)

const (
	YataiApiTokenHeaderName   = "X-YATAI-API-TOKEN"
	NgcOrganizationHeaderName = "Nv-Ngc-Org"
	NgcUserHeaderName         = "Nv-Actor-Id"
)

type DynamoAuthHeaders struct {
	OrgId  string
	UserId string
}

type YataiClient struct {
	endpoint string
	apiToken string
	headers  DynamoAuthHeaders
}

func NewYataiClient(endpoint, apiToken string) *YataiClient {
	return &YataiClient{
		endpoint: endpoint,
		apiToken: apiToken,
	}
}

func (c *YataiClient) SetAuth(headers DynamoAuthHeaders) {
	c.headers = headers
}

func (c *YataiClient) getHeaders() map[string]string {
	return map[string]string{
		YataiApiTokenHeaderName:   c.apiToken,
		NgcOrganizationHeaderName: c.headers.OrgId,
		NgcUserHeaderName:         c.headers.UserId,
	}
}

func (c *YataiClient) GetBento(ctx context.Context, bentoRepositoryName, bentoVersion string) (bento *schemas.DynamoNIM, err error) {
	url_ := urlJoin(c.endpoint, fmt.Sprintf("/api/v1/bento_repositories/%s/bentos/%s", bentoRepositoryName, bentoVersion))
	bento = &schemas.DynamoNIM{}
	_, err = DoJsonRequest(ctx, "GET", url_, c.getHeaders(), nil, nil, bento, nil)
	return
}

func (c *YataiClient) PresignBentoDownloadURL(ctx context.Context, bentoRepositoryName, bentoVersion string) (bento *schemas.DynamoNIM, err error) {
	url_ := urlJoin(c.endpoint, fmt.Sprintf("/api/v1/dynamo_nims/%s/versions/%s/presign_download_url", bentoRepositoryName, bentoVersion))
	bento = &schemas.DynamoNIM{}
	_, err = DoJsonRequest(ctx, "PATCH", url_, c.getHeaders(), nil, nil, bento, nil)
	return
}

func urlJoin(baseURL string, pathPart string) string {
	return strings.TrimRight(baseURL, "/") + "/" + strings.TrimLeft(pathPart, "/")
}
