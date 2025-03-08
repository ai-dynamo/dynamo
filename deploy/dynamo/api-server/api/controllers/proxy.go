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

package controllers

import (
	"net/http"
	"net/http/httputil"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/env"
	"github.com/gin-gonic/gin"
)

type proxyController struct{}

var ProxyController = proxyController{}

func (*proxyController) ReverseProxy(ctx *gin.Context) {
	ndsUrl := env.GetNdsHost()
	director := func(req *http.Request) {
		r := ctx.Request

		req.URL.Scheme = "http"
		req.URL.Host = ndsUrl
		req.Header = r.Header.Clone()
	}
	proxy := &httputil.ReverseProxy{Director: director}
	proxy.ServeHTTP(ctx.Writer, ctx.Request)
}
