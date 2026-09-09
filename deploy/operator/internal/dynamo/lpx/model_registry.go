/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
)

var (
	errRelativeBuildRef  = errors.New("relative ref")
	errBuildFileTooLarge = errors.New("build metadata file exceeds its acquisition limit")
)

const (
	// BuildSchemeFile identifies builds stored on the local filesystem.
	BuildSchemeFile = "file"
	// BuildSchemeGCS identifies builds stored in Google Cloud Storage.
	BuildSchemeGCS = "gs"
)

// ModelRegistry resolves build references and acquires their compiler metadata.
type ModelRegistry struct {
	registryURL *url.URL
	mxClient    modelpb.ModelServiceClient
}

// NewModelRegistry constructs a registry rooted at modelRegistryURL. An empty
// URL is supported for absolute build references. mxClient may be nil when GCS
// downloads are not required.
func NewModelRegistry(modelRegistryURL string, mxClient modelpb.ModelServiceClient) (*ModelRegistry, error) {
	var registryURL *url.URL

	if modelRegistryURL != "" {
		var err error
		registryURL, err = parseBuildRef(modelRegistryURL)
		if err != nil {
			return nil, err
		}
		switch registryURL.Scheme {
		case BuildSchemeFile, BuildSchemeGCS:
		default:
			return nil, fmt.Errorf("unsupported model registry URL scheme %q", registryURL.Scheme)
		}
	}

	return &ModelRegistry{
		registryURL: registryURL,
		mxClient:    mxClient,
	}, nil
}

// EnsureDownloaded ensures buildURL is locally available and reports whether
// the download is complete. For GCS builds it may initiate or advance a
// ModelExpress download. The receiver must be non-nil and is not mutated.
func (r *ModelRegistry) EnsureDownloaded(ctx context.Context, buildURL url.URL) (bool, error) {
	switch buildURL.Scheme {
	case BuildSchemeFile:
		return true, nil
	case BuildSchemeGCS:
		if r.mxClient == nil {
			return false, fmt.Errorf("Model Express client is required for GCS model downloads")
		}

		// Request the GCS build through the current Model Express client and classify its first status.
		rpcCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		modelName := buildURL.String()

		stream, err := r.mxClient.EnsureModelDownloaded(rpcCtx, &modelpb.ModelDownloadRequest{
			ModelName: modelName,
			Provider:  modelpb.ModelProvider_GCS,
		})
		if err != nil {
			return false, fmt.Errorf("ensure model express download for %q: %w", modelName, err)
		}

		update, err := stream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return false, fmt.Errorf("model express returned no status for %q", modelName)
			}
			return false, fmt.Errorf("receive model express download status for %q: %w", modelName, err)
		}

		switch update.GetStatus() {
		case modelpb.ModelStatus_DOWNLOADED:
			return true, nil
		case modelpb.ModelStatus_DOWNLOADING:
			return false, nil
		case modelpb.ModelStatus_ERROR:
			message := update.GetMessage()
			if message == "" {
				message = "model express reported download error"
			}
			return false, fmt.Errorf("model express download failed for %q: %s", modelName, message)
		default:
			return false, fmt.Errorf("model express returned unexpected status %s for %q", update.GetStatus(), modelName)
		}
	default:
		return false, fmt.Errorf("unsupported build download scheme %q", buildURL.Scheme)
	}
}

// BuildURL resolves id against the configured registry URL. The receiver must
// be non-nil and is not mutated.
func (r *ModelRegistry) BuildURL(id string) (*url.URL, error) {
	refURL, err := parseBuildRef(id)
	if err == nil {
		return refURL, nil
	}
	if !errors.Is(err, errRelativeBuildRef) {
		return nil, err
	}

	if r.registryURL == nil {
		return nil, fmt.Errorf("model registry URL is not configured")
	}

	relativeID := strings.TrimSpace(id)
	if !filepath.IsLocal(relativeID) {
		return nil, fmt.Errorf("build ID %q must remain within the configured model registry", id)
	}
	relativeID = filepath.Clean(relativeID)
	if relativeID == "" || relativeID == "." {
		return nil, fmt.Errorf("empty ref")
	}

	if r.registryURL.Scheme == BuildSchemeFile {
		return &url.URL{
			Scheme: BuildSchemeFile,
			Path:   filepath.Clean(filepath.Join(r.registryURL.Path, relativeID)),
		}, nil
	}

	return r.registryURL.JoinPath(relativeID), nil
}

func parseBuildRef(ref string) (*url.URL, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return nil, fmt.Errorf("empty ref")
	}
	cleanRef := filepath.Clean(ref)
	if filepath.IsAbs(cleanRef) {
		return &url.URL{Scheme: BuildSchemeFile, Path: cleanRef}, nil
	}

	uri, err := url.Parse(ref)
	if err != nil {
		return nil, fmt.Errorf("unable to parse ref %q: %w", ref, err)
	}
	if uri.Scheme == "" {
		return nil, fmt.Errorf("ref %q must be an absolute path or URL: %w", ref, errRelativeBuildRef)
	}
	if uri.Scheme == BuildSchemeFile {
		if uri.Host != "" {
			return nil, fmt.Errorf("invalid ref %q: unexpected file host %q", ref, uri.Host)
		}
		if !filepath.IsAbs(uri.Path) {
			return nil, fmt.Errorf("invalid ref %q: file path must be absolute", ref)
		}
		return &url.URL{Scheme: BuildSchemeFile, Path: filepath.Clean(uri.Path)}, nil
	}
	if uri.Host == "" {
		return nil, fmt.Errorf("invalid ref %q: missing host", ref)
	}

	return uri, nil
}

// readBuildFileBounded reads one relative file. The receiver and buildURL must
// be non-nil, and maxBytes must be non-negative.
func (r *ModelRegistry) readBuildFileBounded(
	ctx context.Context,
	buildURL *url.URL,
	relativePath string,
	maxBytes int,
) ([]byte, error) {
	switch buildURL.Scheme {
	case BuildSchemeFile:
		fileURL := buildURL.JoinPath(relativePath)
		file, err := os.Open(filepath.Clean(fileURL.Path))
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", fileURL.String(), err)
		}
		defer func() {
			_ = file.Close()
		}()
		if info, statErr := file.Stat(); statErr == nil && info.Size() > int64(maxBytes) {
			return nil, fmt.Errorf(
				"%w: %s is %d bytes, limit is %d",
				errBuildFileTooLarge,
				fileURL.String(),
				info.Size(),
				maxBytes,
			)
		}
		data, err := io.ReadAll(io.LimitReader(file, int64(maxBytes)+1))
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", fileURL.String(), err)
		}
		if len(data) > maxBytes {
			return nil, fmt.Errorf(
				"%w: %s is larger than %d bytes",
				errBuildFileTooLarge,
				fileURL.String(),
				maxBytes,
			)
		}
		return data, nil
	case BuildSchemeGCS:
		if r.mxClient == nil {
			return nil, fmt.Errorf("Model Express client is required for GCS model registry reads of %q", buildURL.String())
		}

		// Release the metadata stream even when chunk validation stops the read early.
		rpcCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		modelName := buildURL.String()
		stream, err := r.mxClient.StreamModelFiles(rpcCtx, &modelpb.ModelFilesRequest{
			ModelName: modelName,
			Provider:  modelpb.ModelProvider_GCS,
			FileSelector: &modelpb.ModelFileSelector{
				Paths: []string{relativePath},
			},
		})
		if err != nil {
			return nil, fmt.Errorf("streaming %s from %q with Model Express: %w", relativePath, modelName, err)
		}

		var data []byte
		receivedChunk := false
		completedFile := false

		for {
			chunk, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return nil, fmt.Errorf("receiving %s from %q with Model Express: %w", relativePath, modelName, err)
			}
			if completedFile {
				return nil, fmt.Errorf("received extra chunk for %s from %q after final chunk", relativePath, modelName)
			}
			if chunk.GetRelativePath() != relativePath {
				return nil, fmt.Errorf("received unexpected file %q for %s from %q", chunk.GetRelativePath(), relativePath, modelName)
			}
			if chunk.GetOffset() != uint64(len(data)) {
				return nil, fmt.Errorf("received out-of-order chunk for %s from %q: offset %d, want %d", relativePath, modelName, chunk.GetOffset(), len(data))
			}

			receivedChunk = true
			chunkData := chunk.GetData()
			if len(chunkData) > maxBytes-len(data) {
				return nil, fmt.Errorf(
					"%w: %s from %q is larger than %d bytes",
					errBuildFileTooLarge,
					relativePath,
					modelName,
					maxBytes,
				)
			}
			data = append(data, chunkData...)

			if chunk.GetIsLastChunk() {
				completedFile = true
				if totalSize := chunk.GetTotalSize(); totalSize != uint64(len(data)) {
					return nil, fmt.Errorf("received incomplete %s from %q: final size %d, want %d", relativePath, modelName, len(data), totalSize)
				}
			}
		}

		if !receivedChunk {
			return nil, fmt.Errorf("Model Express returned no chunks for %s from %q", relativePath, modelName)
		}
		if !completedFile {
			return nil, fmt.Errorf("Model Express stream for %s from %q ended before final chunk", relativePath, modelName)
		}

		return data, nil
	default:
		return nil, fmt.Errorf("unsupported build metadata scheme %q", buildURL.Scheme)
	}
}

func (r *ModelRegistry) listBuildFiles(ctx context.Context, buildURL *url.URL) ([]string, error) {
	var (
		paths []string
		err   error
	)
	switch buildURL.Scheme {
	case BuildSchemeFile:
		paths, err = localBuildFilePaths(buildURL.Path)
	case BuildSchemeGCS:
		if r.mxClient == nil {
			return nil, fmt.Errorf("Model Express client is required for GCS model registry reads of %q", buildURL.String())
		}

		// List the GCS build files through Model Express before validating the inventory.
		modelName := buildURL.String()
		list, listErr := r.mxClient.ListModelFiles(ctx, &modelpb.ModelFilesRequest{
			ModelName: modelName,
			Provider:  modelpb.ModelProvider_GCS,
		})
		if listErr != nil {
			return nil, fmt.Errorf("listing files from %q with Model Express: %w", modelName, listErr)
		}
		paths = make([]string, 0, len(list.GetFiles()))
		for _, file := range list.GetFiles() {
			if path := file.GetRelativePath(); path != "" {
				paths = append(paths, path)
			}
		}
	default:
		err = fmt.Errorf("unsupported build metadata scheme %q", buildURL.Scheme)
	}
	if err != nil {
		return nil, err
	}
	return normalizeBuildFilePaths(paths)
}

func localBuildFilePaths(root string) ([]string, error) {
	cleanRoot := filepath.Clean(root)
	resolvedRoot, err := filepath.EvalSymlinks(cleanRoot)
	if err != nil {
		return nil, fmt.Errorf("resolving build directory %q: %w", root, err)
	}

	// Keep a malformed manifest directory in the inventory so its required file read fails.
	manifestPath := filepath.Join(resolvedRoot, gbuildManifestV2CapnpFile)
	paths := make([]string, 0)
	err = filepath.WalkDir(resolvedRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && path != manifestPath {
			return nil
		}
		rel, err := filepath.Rel(resolvedRoot, path)
		if err != nil {
			return err
		}
		paths = append(paths, filepath.ToSlash(rel))
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("listing build files under %q: %w", root, err)
	}
	return paths, nil
}
