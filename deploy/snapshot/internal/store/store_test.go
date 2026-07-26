// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package store

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func TestNewReturnsNilForLocalBackends(t *testing.T) {
	for _, backend := range []string{"", "pvc", " pvc "} {
		got, err := New(Config{Backend: backend})
		if err != nil {
			t.Fatalf("New(%q) error: %v", backend, err)
		}
		if got != nil {
			t.Errorf("New(%q) = %v, want nil store", backend, got)
		}
	}
}

func TestNewRejectsUnknownBackend(t *testing.T) {
	if _, err := New(Config{Backend: "gcs"}); err == nil {
		t.Fatal("expected error for unknown backend")
	}
}

func TestNewS3RequiresEndpointBucketCreds(t *testing.T) {
	cases := map[string]S3Config{
		"missing endpoint": {Bucket: "b", AccessKeyID: "a", SecretAccessKey: "s"},
		"missing bucket":   {Endpoint: "localhost:9000", AccessKeyID: "a", SecretAccessKey: "s"},
		"missing creds":    {Endpoint: "localhost:9000", Bucket: "b"},
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := New(Config{Backend: BackendS3, S3: cfg}); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestNewS3StripsSchemeFromEndpoint(t *testing.T) {
	for _, ep := range []string{"https://s3.example.com/", "http://s3.example.com", "s3.example.com"} {
		if _, err := New(Config{Backend: BackendS3, S3: S3Config{
			Endpoint:        ep,
			Bucket:          "checkpoints",
			AccessKeyID:     "key",
			SecretAccessKey: "secret",
		}}); err != nil {
			t.Errorf("New with endpoint %q: %v", ep, err)
		}
	}
}

func TestNormalizePrefix(t *testing.T) {
	cases := map[string]string{
		"":                      "",
		"/":                     "",
		"  ":                    "",
		"ckpt":                  "ckpt/",
		"/ckpt/":                "ckpt/",
		"ckpt/versions/1":       "ckpt/versions/1/",
		"  /ckpt/versions/1/  ": "ckpt/versions/1/",
		"sha256:abc/versions/2": "sha256:abc/versions/2/",
	}
	for in, want := range cases {
		if got := normalizePrefix(in); got != want {
			t.Errorf("normalizePrefix(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestObjectKey(t *testing.T) {
	if got := objectKey("ckpt/versions/1/", filepath.Join("img", "pages-1.img")); got != "ckpt/versions/1/img/pages-1.img" {
		t.Errorf("objectKey = %q", got)
	}
	if got := objectKey("", "manifest.yaml"); got != "manifest.yaml" {
		t.Errorf("objectKey empty prefix = %q", got)
	}
}

func TestSafeDestPath(t *testing.T) {
	dir := t.TempDir()

	ok, err := safeDestPath(dir, "img/pages-1.img")
	if err != nil {
		t.Fatalf("safeDestPath returned error for valid rel: %v", err)
	}
	if !strings.HasPrefix(ok, filepath.Clean(dir)+string(os.PathSeparator)) {
		t.Errorf("dest %q not under %q", ok, dir)
	}

	for _, bad := range []string{"../escape", "img/../../escape", "../../etc/passwd"} {
		if _, err := safeDestPath(dir, bad); err == nil {
			t.Errorf("safeDestPath(%q) should have rejected traversal", bad)
		}
	}
}

// TestS3RoundTrip exercises Upload/Exists/Download/Remove against a live S3
// endpoint. It is skipped unless DYNAMO_TEST_S3_ENDPOINT is set.
//
// Example:
//
//	DYNAMO_TEST_S3_ENDPOINT=localhost:9000 \
//	DYNAMO_TEST_S3_BUCKET=checkpoints \
//	DYNAMO_TEST_S3_ACCESS_KEY=<access-key> \
//	DYNAMO_TEST_S3_SECRET_KEY=<secret-key> \
//	go test ./internal/store/ -run TestS3RoundTrip -v
func TestS3RoundTrip(t *testing.T) {
	endpoint := os.Getenv("DYNAMO_TEST_S3_ENDPOINT")
	if endpoint == "" {
		t.Skip("set DYNAMO_TEST_S3_ENDPOINT to run the S3 round-trip test")
	}
	st, err := New(Config{Backend: BackendS3, S3: S3Config{
		Endpoint:        endpoint,
		Bucket:          getenvDefault("DYNAMO_TEST_S3_BUCKET", "checkpoints"),
		AccessKeyID:     os.Getenv("DYNAMO_TEST_S3_ACCESS_KEY"),
		SecretAccessKey: os.Getenv("DYNAMO_TEST_S3_SECRET_KEY"),
		UseSSL:          os.Getenv("DYNAMO_TEST_S3_SSL") == "1",
		// IP/localhost endpoints need path-style (no bucket DNS).
		ForcePathStyle: os.Getenv("DYNAMO_TEST_S3_FORCE_PATH_STYLE") == "1",
	}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx := context.Background()
	src := t.TempDir()
	writeFile(t, filepath.Join(src, "img", "pages-1.img"), "page-data")
	writeFile(t, filepath.Join(src, "rootfs-diff.tar"), "rootfs")
	writeFile(t, filepath.Join(src, types.ManifestFilename), "checkpointId: sha256:test\n")

	prefix := "dynamo-test/" + t.Name()
	t.Cleanup(func() { _ = st.Remove(context.Background(), prefix) })

	if err := st.Upload(ctx, src, prefix); err != nil {
		t.Fatalf("Upload: %v", err)
	}

	ok, err := st.Exists(ctx, prefix)
	if err != nil || !ok {
		t.Fatalf("Exists after upload = %v, %v; want true, nil", ok, err)
	}

	dst := t.TempDir()
	if err := st.Download(ctx, prefix, dst); err != nil {
		t.Fatalf("Download: %v", err)
	}
	assertFile(t, filepath.Join(dst, "img", "pages-1.img"), "page-data")
	assertFile(t, filepath.Join(dst, types.ManifestFilename), "checkpointId: sha256:test\n")

	if err := st.Remove(ctx, prefix); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	ok, err = st.Exists(ctx, prefix)
	if err != nil {
		t.Fatalf("Exists after remove err: %v", err)
	}
	if ok {
		t.Fatal("Exists after remove = true, want false")
	}
}

func getenvDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func assertFile(t *testing.T, path, want string) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if string(got) != want {
		t.Errorf("%s = %q, want %q", path, got, want)
	}
}
