package reqcli

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

var (
	dftHttpCli  *http.Client
	cliLoadOnce sync.Once
)

var httpTimeout = 90 * time.Second

func getDefaultTransPort() *http.Transport {
	return &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 60 * time.Second,
			DualStack: true,
		}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       180 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		// nolint: gosec
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
}

func GetDefaultHttpClient() *http.Client {
	cliLoadOnce.Do(func() {
		dftHttpCli = &http.Client{
			Timeout:   httpTimeout,
			Transport: getDefaultTransPort(),
		}
	})

	return dftHttpCli
}

func NewHttpCli() (*http.Client, error) {
	return GetDefaultHttpClient(), nil
}

func NewHttpCliWithTimeout(timeout time.Duration) (*http.Client, error) {
	httpCli := &http.Client{
		Timeout:   timeout,
		Transport: getDefaultTransPort(),
	}
	return httpCli, nil
}

type JsonRequestBuilder struct {
	timeout       *time.Duration
	method        string
	url           string
	query         map[string]string
	headers       map[string]string
	payload       interface{}
	result        interface{}
	reqProcessors []func(req *http.Request)
}

func NewJsonRequestBuilder() *JsonRequestBuilder {
	builder := JsonRequestBuilder{}
	return &builder
}

func (b *JsonRequestBuilder) Timeout(timeout time.Duration) *JsonRequestBuilder {
	b.timeout = &timeout
	return b
}

func (b *JsonRequestBuilder) Method(method string) *JsonRequestBuilder {
	b.method = method
	return b
}

func (b *JsonRequestBuilder) Url(url string) *JsonRequestBuilder {
	b.url = url
	return b
}

func (b *JsonRequestBuilder) Query(query map[string]string) *JsonRequestBuilder {
	b.query = query
	return b
}

func (b *JsonRequestBuilder) Headers(headers map[string]string) *JsonRequestBuilder {
	b.headers = headers
	return b
}

func (b *JsonRequestBuilder) Payload(payload interface{}) *JsonRequestBuilder {
	b.payload = payload
	return b
}

func (b *JsonRequestBuilder) Result(result interface{}) *JsonRequestBuilder {
	b.result = result
	return b
}

func (b *JsonRequestBuilder) ProcessReq(processor func(req *http.Request)) *JsonRequestBuilder {
	b.reqProcessors = append(b.reqProcessors, processor)
	return b
}

func (b *JsonRequestBuilder) Do(ctx context.Context) (statusCode int, err error) {
	var req *http.Request

	defer func() {
		if err != nil {
			err = errors.Wrapf(err, "DoJsonRequest Error: [%s]%s", b.method, b.url)
		}
	}()

	if b.payload == nil {
		req, err = http.NewRequestWithContext(ctx, b.method, b.url, nil)
	} else {
		switch p := b.payload.(type) {
		case io.Reader:
			req, err = http.NewRequestWithContext(ctx, b.method, b.url, p)
		default:
			var data []byte
			data, err = json.Marshal(b.payload)
			if err != nil {
				return
			}

			req, err = http.NewRequestWithContext(ctx, b.method, b.url, bytes.NewBuffer(data))
		}
	}

	if err != nil {
		return
	}

	req.Header.Set("Content-Type", "application/json")

	if b.headers != nil {
		for k, v := range b.headers {
			req.Header.Set(k, v)
		}
	}

	for _, reqProcessor := range b.reqProcessors {
		reqProcessor(req)
	}

	q := req.URL.Query()
	for key, value := range b.query {
		q.Add(key, value)
	}
	req.URL.RawQuery = q.Encode()

	cli := GetDefaultHttpClient()
	if b.timeout != nil {
		cli.Timeout = *b.timeout
	}
	logrus.Debugf("http %s %s", b.method, b.url)
	var resp *http.Response
	resp, err = cli.Do(req)
	if err != nil {
		return
	}
	statusCode = resp.StatusCode
	defer func() {
		_ = resp.Body.Close()
	}()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logrus.Errorf("resp.Body error, %s", err)
		return
	}

	if resp.StatusCode != 200 {
		msg := fmt.Sprintf("%s %s status=%d, %s", b.method, b.url, resp.StatusCode, body)
		logrus.Error(msg)
		err = errors.New(msg)
		return
	}

	if b.result != nil {
		err = errors.Wrap(json.Unmarshal(body, b.result), "json unmarshal")
	}

	return
}

func DoJsonRequest(ctx context.Context, method string, url string, headers map[string]string, payload, result interface{}) (err error) {
	_, err = NewJsonRequestBuilder().Method(method).Url(url).Headers(headers).Payload(payload).Result(result).Do(ctx)
	return
}
