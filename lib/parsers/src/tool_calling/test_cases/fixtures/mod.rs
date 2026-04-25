// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! All concrete `ToolCallFixture` impls, one per registered parser.
//!
//! Adding a new parser to the registry must come with a fixture here.
//! The `all_fixtures()` vec is what the contract test in `../tests.rs`
//! iterates over — every parser shows up in the matrix or it's missing.

use super::ToolCallFixture;

mod deepseek_v3;
mod deepseek_v3_1;
mod deepseek_v3_2;
mod deepseek_v4;
mod default;
mod glm47;
mod harmony;
mod hermes;
mod jamba;
mod kimi_k2;
mod llama3_json;
mod minimax_m2;
mod mistral;
mod nemotron_deci;
mod nemotron_nano;
mod phi4;
mod pythonic;
mod qwen3_coder;

pub use deepseek_v3::DeepseekV3Fixture;
pub use deepseek_v3_1::DeepseekV31Fixture;
pub use deepseek_v3_2::DeepseekV32Fixture;
pub use deepseek_v4::DeepseekV4Fixture;
pub use default::DefaultFixture;
pub use glm47::Glm47Fixture;
pub use harmony::HarmonyFixture;
pub use hermes::HermesFixture;
pub use jamba::JambaFixture;
pub use kimi_k2::KimiK2Fixture;
pub use llama3_json::Llama3JsonFixture;
pub use minimax_m2::MinimaxM2Fixture;
pub use mistral::MistralFixture;
pub use nemotron_deci::NemotronDeciFixture;
pub use nemotron_nano::NemotronNanoFixture;
pub use phi4::Phi4Fixture;
pub use pythonic::PythonicFixture;
pub use qwen3_coder::Qwen3CoderFixture;

/// Every parser fixture, in stable alphabetical order by registry name.
/// The matrix report iterates this vec; new parsers go here.
pub fn all_fixtures() -> Vec<Box<dyn ToolCallFixture>> {
    vec![
        Box::new(DefaultFixture),
        Box::new(DeepseekV3Fixture),
        Box::new(DeepseekV31Fixture),
        Box::new(DeepseekV32Fixture),
        Box::new(DeepseekV4Fixture),
        Box::new(Glm47Fixture),
        Box::new(HarmonyFixture),
        Box::new(HermesFixture),
        Box::new(JambaFixture),
        Box::new(KimiK2Fixture),
        Box::new(Llama3JsonFixture),
        Box::new(MinimaxM2Fixture),
        Box::new(MistralFixture),
        Box::new(NemotronDeciFixture),
        Box::new(NemotronNanoFixture),
        Box::new(Phi4Fixture),
        Box::new(PythonicFixture),
        Box::new(Qwen3CoderFixture),
    ]
}
