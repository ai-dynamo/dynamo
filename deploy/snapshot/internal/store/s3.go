// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package store

import (
	"context"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"golang.org/x/sync/errgroup"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const defaultConcurrency = 8

// S3Config configures the S3-backed artifact store. It addresses buckets via
// virtual-hosted (bucket DNS) style by default and uses an explicit endpoint so
// any S3-compatible object store can back checkpoints; no vendor-specific
// features are used.
type S3Config struct {
	// Endpoint is the host:port of the S3 endpoint (scheme optional).
	Endpoint string
	// ForcePathStyle selects path-style addressing (endpoint/bucket/key) instead
	// of the default virtual-hosted style (bucket.endpoint/key). Set this for
	// endpoints that do not support bucket DNS names.
	ForcePathStyle bool
	// Bucket holds the checkpoint artifacts.
	Bucket string
	// Region is the optional region label (defaults server-side).
	Region string
	// AccessKeyID and SecretAccessKey are the static credentials. SessionToken
	// is optional (for temporary credentials).
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	// UseSSL selects https for the endpoint.
	UseSSL bool
	// Concurrency bounds parallel per-object transfers. <= 0 defaults to 8.
	Concurrency int
	// PartSize is the multipart part size in bytes. 0 lets minio-go choose.
	PartSize uint64
}

type s3Store struct {
	client      *minio.Client
	bucket      string
	concurrency int
	partSize    uint64
}

func newS3Store(cfg S3Config) (*s3Store, error) {
	endpoint := strings.TrimSpace(cfg.Endpoint)
	if endpoint == "" {
		return nil, fmt.Errorf("s3 endpoint is required")
	}
	endpoint = strings.TrimPrefix(strings.TrimPrefix(endpoint, "https://"), "http://")
	endpoint = strings.TrimRight(endpoint, "/")

	bucket := strings.TrimSpace(cfg.Bucket)
	if bucket == "" {
		return nil, fmt.Errorf("s3 bucket is required")
	}
	if strings.TrimSpace(cfg.AccessKeyID) == "" || strings.TrimSpace(cfg.SecretAccessKey) == "" {
		return nil, fmt.Errorf("s3 access key id and secret access key are required")
	}

	bucketLookup := minio.BucketLookupDNS // virtual-hosted (bucket DNS) style by default
	if cfg.ForcePathStyle {
		bucketLookup = minio.BucketLookupPath
	}
	client, err := minio.New(endpoint, &minio.Options{
		Creds:        credentials.NewStaticV4(cfg.AccessKeyID, cfg.SecretAccessKey, cfg.SessionToken),
		Secure:       cfg.UseSSL,
		Region:       strings.TrimSpace(cfg.Region),
		BucketLookup: bucketLookup,
	})
	if err != nil {
		return nil, fmt.Errorf("create s3 client for %s: %w", endpoint, err)
	}

	concurrency := cfg.Concurrency
	if concurrency <= 0 {
		concurrency = defaultConcurrency
	}
	return &s3Store{
		client:      client,
		bucket:      bucket,
		concurrency: concurrency,
		partSize:    cfg.PartSize,
	}, nil
}

func (s *s3Store) Backend() string { return BackendS3 }

func (s *s3Store) Upload(ctx context.Context, localDir, keyPrefix string) error {
	prefix := normalizePrefix(keyPrefix)

	var files []string
	manifestSeen := false
	err := filepath.WalkDir(localDir, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		if !d.Type().IsRegular() {
			return fmt.Errorf("checkpoint artifact %s is not a regular file", p)
		}
		rel, err := filepath.Rel(localDir, p)
		if err != nil {
			return err
		}
		if filepath.ToSlash(rel) == types.ManifestFilename {
			manifestSeen = true
			return nil
		}
		files = append(files, rel)
		return nil
	})
	if err != nil {
		return fmt.Errorf("walk checkpoint dir %s: %w", localDir, err)
	}
	if !manifestSeen {
		return fmt.Errorf("checkpoint dir %s has no %s", localDir, types.ManifestFilename)
	}

	// Clear any previous artifact at this prefix before writing. This drops the
	// old manifest first (so Exists reports false for the whole upload window,
	// keeping it all-or-nothing) and removes orphaned objects from a prior
	// artifact that had more/different files.
	if err := s.Remove(ctx, keyPrefix); err != nil {
		return fmt.Errorf("clear existing artifact at s3://%s/%s: %w", s.bucket, prefix, err)
	}

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(s.concurrency)
	for _, rel := range files {
		rel := rel
		g.Go(func() error { return s.uploadOne(gctx, localDir, prefix, rel) })
	}
	if err := g.Wait(); err != nil {
		return err
	}

	// Upload the manifest last: its presence is the completion marker that
	// Exists checks, giving restore an all-or-nothing readiness signal.
	return s.uploadOne(ctx, localDir, prefix, types.ManifestFilename)
}

func (s *s3Store) uploadOne(ctx context.Context, localDir, prefix, rel string) error {
	key := objectKey(prefix, rel)
	src := filepath.Join(localDir, rel)
	opts := minio.PutObjectOptions{ContentType: "application/octet-stream"}
	if s.partSize > 0 {
		opts.PartSize = s.partSize
	}
	if _, err := s.client.FPutObject(ctx, s.bucket, key, src, opts); err != nil {
		return fmt.Errorf("upload %s -> s3://%s/%s: %w", src, s.bucket, key, err)
	}
	return nil
}

func (s *s3Store) Download(ctx context.Context, keyPrefix, localDir string) error {
	prefix := normalizePrefix(keyPrefix)
	if err := os.MkdirAll(localDir, 0o700); err != nil {
		return fmt.Errorf("create restore dir %s: %w", localDir, err)
	}

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(s.concurrency)
	found := false
	for obj := range s.client.ListObjects(gctx, s.bucket, minio.ListObjectsOptions{Prefix: prefix, Recursive: true}) {
		if obj.Err != nil {
			_ = g.Wait()
			return fmt.Errorf("list s3://%s/%s: %w", s.bucket, prefix, obj.Err)
		}
		rel := strings.TrimPrefix(obj.Key, prefix)
		if rel == "" || strings.HasSuffix(rel, "/") {
			continue
		}
		found = true
		key := obj.Key
		g.Go(func() error { return s.downloadOne(gctx, key, rel, localDir) })
	}
	if err := g.Wait(); err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("no objects under s3://%s/%s", s.bucket, prefix)
	}
	return nil
}

func (s *s3Store) downloadOne(ctx context.Context, key, rel, localDir string) error {
	dest, err := safeDestPath(localDir, rel)
	if err != nil {
		return fmt.Errorf("object key %q: %w", key, err)
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o700); err != nil {
		return fmt.Errorf("create dir for %s: %w", dest, err)
	}
	if err := s.client.FGetObject(ctx, s.bucket, key, dest, minio.GetObjectOptions{}); err != nil {
		return fmt.Errorf("download s3://%s/%s -> %s: %w", s.bucket, key, dest, err)
	}
	return nil
}

func (s *s3Store) Exists(ctx context.Context, keyPrefix string) (bool, error) {
	prefix := normalizePrefix(keyPrefix)
	manifestKey := prefix + types.ManifestFilename
	if _, err := s.client.StatObject(ctx, s.bucket, manifestKey, minio.StatObjectOptions{}); err != nil {
		if resp := minio.ToErrorResponse(err); resp.StatusCode == http.StatusNotFound {
			return false, nil
		}
		return false, fmt.Errorf("stat s3://%s/%s: %w", s.bucket, manifestKey, err)
	}
	return true, nil
}

func (s *s3Store) Remove(ctx context.Context, keyPrefix string) error {
	prefix := normalizePrefix(keyPrefix)
	objCh := make(chan minio.ObjectInfo)
	var listErr error
	go func() {
		defer close(objCh)
		// Per the minio-go ListObjects contract, obj.Err signals a listing
		// failure; stop feeding RemoveObjects and surface it (don't skip).
		for obj := range s.client.ListObjects(ctx, s.bucket, minio.ListObjectsOptions{Prefix: prefix, Recursive: true}) {
			if obj.Err != nil {
				listErr = obj.Err
				return
			}
			objCh <- obj
		}
	}()
	// Drain RemoveObjects fully so the producer never blocks; keep the first
	// removal error.
	var removeErr error
	for rerr := range s.client.RemoveObjects(ctx, s.bucket, objCh, minio.RemoveObjectsOptions{}) {
		if rerr.Err != nil && removeErr == nil {
			removeErr = fmt.Errorf("remove s3://%s/%s: %w", s.bucket, rerr.ObjectName, rerr.Err)
		}
	}
	if removeErr != nil {
		return removeErr
	}
	if listErr != nil {
		return fmt.Errorf("list s3://%s/%s for removal: %w", s.bucket, prefix, listErr)
	}
	return nil
}

// objectKey joins a normalized prefix with a relative path, converting OS path
// separators to forward slashes.
func objectKey(prefix, rel string) string {
	return prefix + filepath.ToSlash(rel)
}

// safeDestPath resolves rel under localDir and rejects keys that would escape it
// (path traversal from crafted object keys).
func safeDestPath(localDir, rel string) (string, error) {
	cleanDir := filepath.Clean(localDir)
	dest := filepath.Join(cleanDir, filepath.FromSlash(rel))
	if dest != cleanDir && !strings.HasPrefix(dest, cleanDir+string(os.PathSeparator)) {
		return "", fmt.Errorf("escapes restore dir %s", cleanDir)
	}
	return dest, nil
}
