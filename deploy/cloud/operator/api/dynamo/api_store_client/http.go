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

package api_store_client

import (
	"context"
	"crypto/tls"
	"fmt"
	"time"

	"resty.dev/v3"
)

var defaultClient *resty.Client

func GetDefaultClient() *resty.Client {
	if defaultClient == nil {
		defaultClient = resty.New().
			SetTimeout(90*time.Second).
			SetRetryCount(3).
			SetRetryWaitTime(2*time.Second).
			SetRetryMaxWaitTime(10*time.Second).
			SetHeader("Content-Type", "application/json").
			SetTLSClientConfig(&tls.Config{InsecureSkipVerify: true}) // Optional: mirrors your custom transport
	}
	return defaultClient
}

func DoJsonRequest(ctx context.Context, method string, url string, headers map[string]string, query map[string]string, payload interface{}, result interface{}, timeout *time.Duration) (int, error) {
	client := GetDefaultClient()

	if timeout != nil {
		client.SetTimeout(*timeout)
	}

	req := client.R().
		SetContext(ctx).
		SetBody(payload).
		SetResult(result).
		SetHeaders(headers).
		SetQueryParams(query)

	var resp *resty.Response
	var err error

	switch method {
	case "GET":
		resp, err = req.Get(url)
	case "POST":
		resp, err = req.Post(url)
	case "PUT":
		resp, err = req.Put(url)
	case "DELETE":
		resp, err = req.Delete(url)
	case "PATCH":
		resp, err = req.Patch(url)
	default:
		return 0, fmt.Errorf("unsupported method: %s", method)
	}

	if err != nil {
		return 0, fmt.Errorf("request error: %w", err)
	}

	if resp.IsError() {
		return resp.StatusCode(), fmt.Errorf("http %s %s failed with status %d: %s", method, url, resp.StatusCode(), resp.String())
	}

	return resp.StatusCode(), nil
}
