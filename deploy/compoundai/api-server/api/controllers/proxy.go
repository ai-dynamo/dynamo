package controllers

import (
	"net/http"
	"net/http/httputil"

	"github.com/gin-gonic/gin"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/env"
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
