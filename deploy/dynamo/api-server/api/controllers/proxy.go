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
	"net/url"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/env"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

type proxyController struct{}

var ProxyController = proxyController{}

func (*proxyController) ReverseProxy(ctx *gin.Context) {
	backendUrl := env.GetBackendUrl()
	target, err := url.Parse(backendUrl)
	if err != nil {
		ctx.AbortWithStatus(http.StatusInternalServerError)
		return
	}

	log.Info().Msgf("Proxying request to: %s%s", backendUrl, ctx.Request.URL.Path)

	director := func(req *http.Request) {
		r := ctx.Request
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.URL.Path = r.URL.Path
		req.URL.RawQuery = r.URL.RawQuery
		req.Header = r.Header.Clone()

		req.Host = target.Host
	}

	proxy := &httputil.ReverseProxy{
		Director: director,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			log.Error().Msgf("Proxy error: %v", err)
			ctx.AbortWithStatus(http.StatusBadGateway)
		},
	}

	proxy.ServeHTTP(ctx.Writer, ctx.Request)
}
