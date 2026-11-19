// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stand-in for the `symphonia` facade crate.
//!
//! It provides the API that llm-multimodal uses: `core` and the `default::get_codecs`
//! and `default::get_probe` registries. The registries hold the codecs, formats and
//! metadata readers of Symphonia's `all` feature set, except AAC.

#![forbid(unsafe_code)]

pub use symphonia_core as core;

pub mod default {
    use std::sync::LazyLock;

    use symphonia_core::codecs::registry::CodecRegistry;
    use symphonia_core::formats::probe::Probe;

    static CODECS: LazyLock<CodecRegistry> = LazyLock::new(|| {
        let mut registry = CodecRegistry::new();
        registry.register_audio_decoder::<symphonia_codec_adpcm::AdpcmDecoder>();
        registry.register_audio_decoder::<symphonia_codec_alac::AlacDecoder>();
        registry.register_audio_decoder::<symphonia_bundle_flac::FlacDecoder>();
        registry.register_audio_decoder::<symphonia_bundle_mp3::MpaDecoder>();
        registry.register_audio_decoder::<symphonia_codec_pcm::PcmDecoder>();
        registry.register_audio_decoder::<symphonia_codec_vorbis::VorbisDecoder>();
        registry
    });

    // Same order as the upstream facade, because the probe tries formats in order.
    static PROBE: LazyLock<Probe> = LazyLock::new(|| {
        let mut probe = Probe::default();
        probe.register_format::<symphonia_format_caf::CafReader<'_>>();
        probe.register_format::<symphonia_bundle_flac::FlacReader<'_>>();
        probe.register_format::<symphonia_format_isomp4::IsoMp4Reader<'_>>();
        probe.register_format::<symphonia_bundle_mp3::MpaReader<'_>>();
        probe.register_format::<symphonia_format_riff::AiffReader<'_>>();
        probe.register_format::<symphonia_format_riff::WavReader<'_>>();
        probe.register_format::<symphonia_format_ogg::OggReader<'_>>();
        probe.register_format::<symphonia_format_mkv::MkvReader<'_>>();
        probe.register_metadata::<symphonia_metadata::ape::ApeReader<'_>>();
        probe.register_metadata::<symphonia_metadata::id3v1::Id3v1Reader<'_>>();
        probe.register_metadata::<symphonia_metadata::id3v2::Id3v2Reader<'_>>();
        probe
    });

    /// Returns the registry of all enabled audio decoders.
    pub fn get_codecs() -> &'static CodecRegistry {
        &CODECS
    }

    /// Returns the probe with all enabled format and metadata readers.
    pub fn get_probe() -> &'static Probe {
        &PROBE
    }
}
