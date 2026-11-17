/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	modelpb "github.com/ai-dynamo/modelexpress/modelexpress_client/go/gen/modelexpress/model"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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

type defaultModelRegistry struct {
	registryURL *url.URL
	mxClient    modelpb.ModelServiceClient
}

// ModelRegistry resolves build references, acquires their compiler metadata,
// and ensures builds and checkpoints are downloaded.
type ModelRegistry interface {
	BuildURL(id string) (*url.URL, error)
	EnsureDownloaded(ctx context.Context, buildURL url.URL) (bool, error)
	EnsureCheckpointDownloaded(ctx context.Context, checkpoint *Checkpoint) (bool, error)
	AcquireBuild(ctx context.Context, id string) (*Build, error)
}

// NewModelRegistry constructs a registry rooted at modelRegistryURL. An empty
// URL is supported for absolute build references. mxClient may be nil when GCS
// build and checkpoint downloads are not required.
func NewModelRegistry(modelRegistryURL string, mxClient modelpb.ModelServiceClient) (ModelRegistry, error) {
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

	return &defaultModelRegistry{
		registryURL: registryURL,
		mxClient:    mxClient,
	}, nil
}

// EnsureDownloaded ensures buildURL is locally available and reports whether
// the download is complete. For GCS builds it may initiate or advance a
// ModelExpress download. The receiver must be non-nil and is not mutated.
func (r *defaultModelRegistry) EnsureDownloaded(ctx context.Context, buildURL url.URL) (bool, error) {
	switch buildURL.Scheme {
	case BuildSchemeFile:
		return true, nil
	case BuildSchemeGCS:
		if r.mxClient == nil {
			return false, fmt.Errorf("Model Express client is required for GCS model downloads")
		}
		downloaded, _, err := r.ensureModelExpressDownload(ctx, &modelpb.ModelDownloadRequest{
			ModelName: buildURL.String(),
			Provider:  modelpb.ModelProvider_GCS,
		})
		return downloaded, err
	default:
		return false, fmt.Errorf("unsupported build download scheme %q", buildURL.Scheme)
	}
}

// EnsureCheckpointDownloaded ensures the non-nil validated checkpoint is in model
// storage and reports whether the download is complete. It may initiate or
// advance a Model Express download pinned to the checkpoint revision, and fails
// unless Model Express confirms that exact revision. The receiver must be
// non-nil and is not mutated.
func (r *defaultModelRegistry) EnsureCheckpointDownloaded(
	ctx context.Context,
	checkpoint *Checkpoint,
) (bool, error) {
	if r.mxClient == nil {
		return false, fmt.Errorf("checkpoint %q requires Model Express; configure infrastructure.modelExpressURL", CheckpointKey(checkpoint))
	}

	// Request the pinned revision; Model Express resolves it before claiming a download.
	downloaded, update, err := r.ensureModelExpressDownload(ctx, &modelpb.ModelDownloadRequest{
		ModelName: checkpoint.Model,
		Provider:  modelpb.ModelProvider_HUGGING_FACE,
		Revision:  &checkpoint.Revision,
	})
	if err != nil {
		return false, err
	}

	// Servers before Model Express 0.6.0 ignore the revision and report none.
	if update.ResolvedRevision == nil {
		return false, fmt.Errorf(
			"model express did not report a resolved revision for %q; checkpoint downloads require Model Express 0.6.0 or later",
			CheckpointKey(checkpoint),
		)
	}
	if resolved := update.GetResolvedRevision(); resolved != checkpoint.Revision {
		return false, fmt.Errorf("model express resolved %q to revision %q, want %q", checkpoint.Model, resolved, checkpoint.Revision)
	}
	return downloaded, nil
}

// ensureModelExpressDownload sends one download request and classifies the first
// status update. It returns the update when the status is DOWNLOADED or
// DOWNLOADING. The receiver must have a non-nil Model Express client.
func (r *defaultModelRegistry) ensureModelExpressDownload(
	ctx context.Context,
	request *modelpb.ModelDownloadRequest,
) (bool, *modelpb.ModelStatusUpdate, error) {
	rpcCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	modelName := request.GetModelName()

	stream, err := r.mxClient.EnsureModelDownloaded(rpcCtx, request)
	if err != nil {
		return false, nil, fmt.Errorf("ensure model express download for %q: %w", modelName, err)
	}

	update, err := stream.Recv()
	if err != nil {
		if errors.Is(err, io.EOF) {
			return false, nil, fmt.Errorf("model express returned no status for %q", modelName)
		}
		return false, nil, fmt.Errorf("receive model express download status for %q: %w", modelName, err)
	}

	switch update.GetStatus() {
	case modelpb.ModelStatus_DOWNLOADED:
		return true, update, nil
	case modelpb.ModelStatus_DOWNLOADING:
		return false, update, nil
	case modelpb.ModelStatus_ERROR:
		message := update.GetMessage()
		if message == "" {
			message = "model express reported download error"
		}
		return false, nil, fmt.Errorf("model express download failed for %q: %s", modelName, message)
	default:
		return false, nil, fmt.Errorf("model express returned unexpected status %s for %q", update.GetStatus(), modelName)
	}
}

// BuildURL resolves id against the configured registry URL. Absolute references
// require an unconfigured registry. The receiver must be non-nil and is not mutated.
func (r *defaultModelRegistry) BuildURL(id string) (*url.URL, error) {
	refURL, err := parseBuildRef(id)
	if err == nil {
		if r.registryURL != nil {
			return nil, fmt.Errorf("build ID %q must be relative when model registry URL is configured", id)
		}
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
	if relativeID == "." {
		return nil, fmt.Errorf("empty ref")
	}

	if r.registryURL.Scheme == BuildSchemeFile {
		return &url.URL{
			Scheme: BuildSchemeFile,
			Path:   filepath.Join(r.registryURL.Path, relativeID),
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
// A missing remote file wraps os.ErrNotExist only before any chunks arrive.
func (r *defaultModelRegistry) readBuildFileBounded(
	ctx context.Context,
	buildURL *url.URL,
	relativePath string,
	maxBytes int,
) ([]byte, error) {
	switch buildURL.Scheme {
	case BuildSchemeFile:
		// Open nonblocking so a FIFO cannot stall before file-type validation.
		fileURL := buildURL.JoinPath(relativePath)
		file, err := os.OpenFile(filepath.Clean(fileURL.Path), os.O_RDONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", fileURL.String(), err)
		}
		defer func() {
			_ = file.Close()
		}()

		// Validate the opened file, not a path that could be replaced.
		info, err := file.Stat()
		if err != nil {
			return nil, fmt.Errorf("checking %s: %w", fileURL.String(), err)
		}
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("reading %s: not a regular file", fileURL.String())
		}
		if info.Size() > int64(maxBytes) {
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
			if status.Code(err) == codes.NotFound {
				err = fmt.Errorf("%w: %w", os.ErrNotExist, err)
			}
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
				if !receivedChunk && status.Code(err) == codes.NotFound {
					err = fmt.Errorf("%w: %w", os.ErrNotExist, err)
				}
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

var errInvalidBuildManifest = errors.New("invalid build manifest")

const (
	// This domain identifies snapshots of manifest bytes without a file inventory.
	buildSnapshotIdentityVersion = "dynamo-lpx-compiler-snapshot/v4"
	// Bound the single manifest to the Model Express transport's metadata ceiling.
	maxBuildSnapshotMetadataBytes = modelExpressMaxMessageSize
	buildSnapshotTimeout          = 30 * time.Second
)

// AcquireBuild reads and validates the immutable compiler manifest once.
// The receiver must be non-nil and is not mutated. Payload files are not inspected.
func (r *defaultModelRegistry) AcquireBuild(ctx context.Context, id string) (*Build, error) {
	ctx, cancel := context.WithTimeout(ctx, buildSnapshotTimeout)
	defer cancel()
	refURL, err := r.BuildURL(id)
	if err != nil {
		return nil, err
	}

	format, manifestData, err := r.readBuildContract(ctx, refURL)
	if err != nil {
		return nil, err
	}
	manifestPath, err := format.filename()
	if err != nil {
		return nil, err
	}

	// Validate the selected contract exactly once, without fallback on invalid split bytes.
	var build *Build
	switch format {
	case buildContractManifestV2:
		manifest, decodeErr := decodeGbuildManifestV2(manifestData)
		if decodeErr != nil {
			err = decodeErr
		} else {
			build, err = buildFromGbuildManifestV2(refURL.String(), manifest)
		}
	case buildContractDeploymentV1:
		deployment, decodeErr := decodeGbuildDeploymentV1(manifestData)
		if decodeErr != nil {
			err = decodeErr
		} else {
			build, err = buildFromGbuildDeploymentV1(refURL.String(), deployment)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("%w for %q: %w", errInvalidBuildManifest, refURL.String(), err)
	}

	// Hash the exact manifest bytes in their own domain, bound to the schema filename and independent of the locator.
	hash := sha256.New()
	_, _ = hash.Write([]byte(buildSnapshotIdentityVersion + "\x00"))
	_, _ = hash.Write([]byte(manifestPath + "\x00"))
	_, _ = hash.Write(manifestData)
	build.contentID = fmt.Sprintf("sha256:%x", hash.Sum(nil))
	build.contractFormat = format
	return build, nil
}

// readBuildContract selects one supported entry point and returns its bounded bytes.
// The receiver and buildURL must be non-nil.
func (r *defaultModelRegistry) readBuildContract(ctx context.Context, buildURL *url.URL) (buildContractFormat, []byte, error) {
	candidates := []string{gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile}
	paths := make([]string, 0, len(candidates))
	var manifestData []byte
	for _, name := range candidates {
		if buildURL.Scheme == BuildSchemeFile {
			// Inspect local entries first so broken symlinks still count as published contracts.
			if err := ctx.Err(); err != nil {
				return buildContractUnknown, nil, err
			}
			_, err := os.Lstat(buildURL.JoinPath(name).Path)
			if errors.Is(err, os.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
				continue
			}
			if err != nil {
				return buildContractUnknown, nil, err
			}
		} else {
			// Read alternatives separately because Model Express requires every selected path to exist.
			data, err := r.readBuildFileBounded(ctx, buildURL, name, maxBuildSnapshotMetadataBytes)
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			if err != nil {
				return buildContractUnknown, nil, err
			}
			manifestData = data
		}
		paths = append(paths, name)
	}

	if len(paths) == 0 {
		return buildContractUnknown, nil, fmt.Errorf("%w: missing build contract: require %s or %s", os.ErrNotExist, gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile)
	}
	format, err := selectBuildContract(paths)
	if err != nil {
		return buildContractUnknown, nil, fmt.Errorf("%w: %w", errInvalidBuildManifest, err)
	}
	if buildURL.Scheme == BuildSchemeFile {
		manifestData, err = r.readBuildFileBounded(ctx, buildURL, paths[0], maxBuildSnapshotMetadataBytes)
		if err != nil {
			return buildContractUnknown, nil, err
		}
	}
	return format, manifestData, nil
}
