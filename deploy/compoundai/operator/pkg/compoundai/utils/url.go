package utils

import (
	"net/url"
	"path"
)

func UrlJoin(baseUrl string, extra string, params ...map[string]string) string {
	u, err := url.Parse(baseUrl)
	if err != nil {
		return baseUrl
	}
	u.Path = path.Join(u.Path, extra)
	q := u.Query()
	for _, p := range params {
		for k, v := range p {
			q.Add(k, v)
		}
	}
	u.RawQuery = q.Encode()
	return u.String()
}

func UrlJoinWithQuery(baseUrl string, extra string, query url.Values) string {
	u, err := url.Parse(baseUrl)
	if err != nil {
		return baseUrl
	}
	u.Path = path.Join(u.Path, extra)
	u.RawQuery = query.Encode()
	return u.String()
}
