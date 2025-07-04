package common

import (
	"fmt"
	"net/url"
	"strings"
)

func GetHost(someURL string) (string, error) {
	// Add scheme if not present
	if !strings.Contains(someURL, "://") {
		someURL = "dummy://" + someURL
	}
	url, err := url.Parse(someURL)
	if err != nil {
		return "", err
	}
	if url.Host == "" {
		return "", fmt.Errorf("no host found in URL %q", someURL)
	}
	return url.Host, nil
}
