/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash"
)

const (
	// Preserve the versioned digest domains so this terminology refactor does not
	// introduce an additional identity change beyond the base PR.
	modelProjectionDigestVersion = "dynamo-node-local-compilation/v1"
	workloadSetDigestVersion     = "dynamo-lpx-compilation-set/v1"
	maxLPXPartitions             = 256
)

// appendModelProjections appends one component's immutable model projections to
// the caller-owned destination, which may be nil. Existing elements are unchanged.
// intent.BuildSnapshot contains a normalized, non-nil build; Models is nonempty
// and Pipeline is selected by the validated resolver. ModelSettings is API-validated
// JSON: omitted, null or an object. Source inputs are not mutated; discard error results.
func appendModelProjections(dst []*ModelProjection, intent ModelProjectionInput) ([]*ModelProjection, error) {
	modelSettings, modelSettingsObject := canonicalModelSettings(intent.ModelSettings)
	intent.ModelSettings = modelSettings

	var projections []*ModelProjection
	var err error
	switch intent.BuildSnapshot.build.Family {
	case BuildFamilyXT:
		projections, err = appendV2ModelProjections(dst, intent, modelSettingsObject)
	case BuildFamilyHX:
		projections, err = appendV3ModelProjections(dst, intent, modelSettingsObject)
	default:
		return nil, fmt.Errorf("unsupported LPX target family %q", intent.BuildSnapshot.build.Family)
	}
	if err != nil {
		return nil, err
	}
	// Bound the component's physical build before runtime expansion and request publication.
	projection := projections[len(dst)]
	if len(projection.partitions) < 1 || len(projection.partitions) > maxLPXPartitions {
		return nil, fmt.Errorf("LPX projection has %d partitions, limit is 1..%d", len(projection.partitions), maxLPXPartitions)
	}
	return projections, nil
}

func workloadSetDigest(projections []*ModelProjection) (WorkloadDigest, error) {
	first := projections[0]
	if len(projections) == 1 {
		return first.digest, nil
	}
	transcript := &digestTranscript{hash: sha256.New()}
	transcript.field("schema", []byte(workloadSetDigestVersion))
	for _, projection := range projections {
		if projection.configuredBuild.Family != first.configuredBuild.Family {
			return WorkloadDigest{}, fmt.Errorf(
				"LPX model projections have mixed target families %q and %q",
				first.configuredBuild.Family,
				projection.configuredBuild.Family,
			)
		}
		transcript.field("model", []byte(projection.model))
		// Keep the versioned field tag stable so this terminology refactor does not
		// introduce an additional aggregate identity change beyond the base PR.
		transcript.field("compilation-digest", projection.digest[:])
	}
	return transcript.sum(), nil
}

// canonicalModelSettings canonicalizes an API-validated optional JSON object.
func canonicalModelSettings(raw json.RawMessage) (json.RawMessage, map[string]any) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, make(map[string]any)
	}

	// Decode the admitted object while preserving number spellings for identity.
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	decoder.UseNumber()
	var object map[string]any
	_ = decoder.Decode(&object)

	// Decoder-owned values encode infallibly in memory; drop Encoder's documented trailing newline.
	var output bytes.Buffer
	encoder := json.NewEncoder(&output)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(object)
	return output.Bytes()[:output.Len()-1], object
}

func newModelProjectionTranscripts(intent ModelProjectionInput, projectionVersion string) []digestTranscript {
	// Each logical model owns a fresh hash while sharing the component's canonical inputs.
	transcripts := make([]digestTranscript, len(intent.Models))
	for index, model := range intent.Models {
		transcript := &transcripts[index]
		transcript.hash = sha256.New()
		transcript.field("schema", []byte(modelProjectionDigestVersion))
		// Keep the versioned field tag stable so this terminology refactor does not
		// introduce an additional identity change beyond the base PR.
		transcript.field("lowerer", []byte(projectionVersion))
		// Ref is an acquisition locator, not build content. In particular, a
		// file-backed snapshot's ref contains its absolute checkout path.
		transcript.field("build-content-id", []byte(intent.BuildSnapshot.contentID))
		// Bind the family derived from immutable build metadata without retaining the
		// removed API selector.
		version := "v2"
		if intent.BuildSnapshot.build.Family == BuildFamilyHX {
			version = "v3"
		}
		transcript.field("device-version", []byte(version))
		transcript.field("pipeline", []byte(intent.Pipeline))

		// Preserve the digest's wire values without storing a second runtime discriminator.
		mode := "lpuOnly"
		if intent.Pipeline == PipelineLPX {
			mode = "strictHybrid"
		}
		transcript.field("workload-mode", []byte(mode))

		transcript.field("model", []byte(model))
		transcript.field("model-settings", intent.ModelSettings)
	}
	return transcripts
}

// bindHybridRuntimeIO records the shared Cyborg I/O contract in a model projection digest.
func bindHybridRuntimeIO(
	transcript *digestTranscript,
	pipeline Pipeline,
	ioFPGACount int32,
	ioFanoutFactor int32,
) {
	if pipeline != PipelineLPX {
		return
	}

	transcript.intField("io-fpga-count", int64(ioFPGACount))
	if ioFanoutFactor > 1 {
		transcript.intField("io-fanout-factor", int64(ioFanoutFactor))
	}
}

type digestTranscript struct {
	hash hash.Hash
}

func (t *digestTranscript) field(tag string, value []byte) {
	writeLengthDelimited(t.hash, []byte(tag))
	writeLengthDelimited(t.hash, value)
}

func (t *digestTranscript) uint32Field(tag string, value uint32) {
	var encoded [4]byte
	binary.BigEndian.PutUint32(encoded[:], value)
	t.field(tag, encoded[:])
}

func (t *digestTranscript) intField(tag string, value int64) {
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], uint64(value))
	t.field(tag, encoded[:])
}

func (t *digestTranscript) sum() WorkloadDigest {
	var digest WorkloadDigest
	copy(digest[:], t.hash.Sum(nil))
	return digest
}

func writeLengthDelimited(writer hash.Hash, value []byte) {
	var length [8]byte
	binary.BigEndian.PutUint64(length[:], uint64(len(value)))
	_, _ = writer.Write(length[:])
	_, _ = writer.Write(value)
}
