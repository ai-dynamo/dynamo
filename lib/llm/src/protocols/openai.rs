// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{
    ContentProvider,
    common::{self, OutputOptionsProvider, SamplingOptionsProvider, StopConditionsProvider},
};
use crate::protocols::openai::common_ext::CommonExtProvider;

pub mod chat_completions;
pub mod common_ext;
pub mod completions;
pub mod embeddings;
pub mod models;
pub mod nvext;
pub mod responses;
pub mod tools;
pub mod validate;

use validate::{
    BEST_OF_RANGE, FREQUENCY_PENALTY_RANGE, MIN_P_RANGE, N_RANGE, PRESENCE_PENALTY_RANGE,
    TEMPERATURE_RANGE, TOP_P_RANGE, validate_range,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct AnnotatedDelta<R> {
    pub delta: R,
    pub id: Option<String>,
    pub event: Option<String>,
    pub comment: Option<String>,
}

trait OpenAISamplingOptionsProvider {
    fn get_temperature(&self) -> Option<f32>;

    fn get_top_p(&self) -> Option<f32>;

    fn get_frequency_penalty(&self) -> Option<f32>;

    fn get_presence_penalty(&self) -> Option<f32>;

    fn get_seed(&self) -> Option<i64>;

    fn get_n(&self) -> Option<u8>;

    fn get_best_of(&self) -> Option<u8>;

    fn nvext(&self) -> Option<&nvext::NvExt>;
}

trait OpenAIStopConditionsProvider {
    fn get_max_tokens(&self) -> Option<u32>;

    fn get_min_tokens(&self) -> Option<u32>;

    fn get_stop(&self) -> Option<Vec<String>>;

    fn nvext(&self) -> Option<&nvext::NvExt>;

    /// Get ignore_eos from CommonExt if the type supports it.
    /// Default returns None for types without CommonExt support.
    fn get_common_ignore_eos(&self) -> Option<bool> {
        None
    }

    /// Get the effective ignore_eos value from CommonExt.
    fn get_ignore_eos(&self) -> Option<bool> {
        self.get_common_ignore_eos()
    }

    /// Get max_thinking_tokens from nvext
    /// NOTE: This is currently a passthrough for future thinking budget implementation
    fn get_max_thinking_tokens(&self) -> Option<u32> {
        self.nvext().and_then(|nv| nv.max_thinking_tokens)
    }
}

trait OpenAIOutputOptionsProvider {
    fn get_logprobs(&self) -> Option<u32>;

    fn get_prompt_logprobs(&self) -> Option<u32>;

    fn get_skip_special_tokens(&self) -> Option<bool>;

    fn get_formatted_prompt(&self) -> Option<bool>;
}

impl<T: OpenAISamplingOptionsProvider + CommonExtProvider> SamplingOptionsProvider for T {
    fn extract_sampling_options(&self) -> Result<common::SamplingOptions> {
        // let result = self.validate();
        // if let Err(e) = result {
        //     return Err(format!("Error validating sampling options: {}", e));
        // }

        let mut temperature = validate_range(self.get_temperature(), &TEMPERATURE_RANGE)
            .map_err(|e| anyhow::anyhow!("Error validating temperature: {}", e))?;
        let mut top_p = validate_range(self.get_top_p(), &TOP_P_RANGE)
            .map_err(|e| anyhow::anyhow!("Error validating top_p: {}", e))?;
        let frequency_penalty =
            validate_range(self.get_frequency_penalty(), &FREQUENCY_PENALTY_RANGE)
                .map_err(|e| anyhow::anyhow!("Error validating frequency_penalty: {}", e))?;
        let presence_penalty = validate_range(self.get_presence_penalty(), &PRESENCE_PENALTY_RANGE)
            .map_err(|e| anyhow::anyhow!("Error validating presence_penalty: {}", e))?;
        let top_k = CommonExtProvider::get_top_k(self);
        let repetition_penalty = CommonExtProvider::get_repetition_penalty(self);
        let include_stop_str_in_output = CommonExtProvider::get_include_stop_str_in_output(self);
        let seed = self.get_seed();
        let n = validate_range(self.get_n(), &N_RANGE)
            .map_err(|e| anyhow::anyhow!("Error validating n: {}", e))?;
        let best_of = validate_range(self.get_best_of(), &BEST_OF_RANGE)
            .map_err(|e| anyhow::anyhow!("Error validating best_of: {}", e))?;

        let min_p = validate_range(CommonExtProvider::get_min_p(self), &MIN_P_RANGE)
            .map_err(|e| anyhow::anyhow!("Error validating min_p: {}", e))?;

        if let Some(nvext) = self.nvext() {
            let greedy = nvext.greed_sampling.unwrap_or(false);
            if greedy {
                top_p = None;
                temperature = None;
            }
        }

        let guided_decoding_backend = self.get_guided_decoding_backend();
        let guided_json = self.get_guided_json();
        let guided_regex = self.get_guided_regex();
        let guided_grammar = self.get_guided_grammar();
        let guided_choice = self.get_guided_choice();
        let guided_whitespace_pattern = self.get_guided_whitespace_pattern();

        let guided_decoding = match common::GuidedDecodingOptions::from_optional(
            guided_json,
            guided_regex,
            guided_choice,
            guided_grammar,
            guided_decoding_backend,
            guided_whitespace_pattern,
        ) {
            Ok(options) => options,
            Err(e) => {
                // Handle the validation error (log, return error, etc.)
                tracing::error!("Invalid guided decoding options: {:?}", e);
                return Err(e);
            }
        };

        Ok(common::SamplingOptions {
            n,
            best_of,
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
            temperature,
            top_p,
            top_k,
            min_p,
            seed,
            use_beam_search: None,
            length_penalty: None,
            guided_decoding,
            include_stop_str_in_output,
        })
    }
}

impl<T: OpenAIStopConditionsProvider> StopConditionsProvider for T {
    fn extract_stop_conditions(&self) -> Result<common::StopConditions> {
        let max_tokens = self.get_max_tokens();
        let min_tokens = self.get_min_tokens();
        let stop = self.get_stop();
        let max_thinking_tokens = self.get_max_thinking_tokens();

        if let Some(stop) = &stop
            && stop.len() > 4
        {
            anyhow::bail!("stop conditions must be less than 4")
        }

        // Use the trait method to get ignore_eos, which handles precedence
        let ignore_eos = self.get_ignore_eos();

        Ok(common::StopConditions {
            max_tokens,
            min_tokens,
            stop,
            stop_token_ids_hidden: None,
            ignore_eos,
            max_thinking_tokens,
        })
    }
}

impl<T: OpenAIOutputOptionsProvider> OutputOptionsProvider for T {
    fn extract_output_options(&self) -> Result<common::OutputOptions> {
        let logprobs = self.get_logprobs();
        let prompt_logprobs = self.get_prompt_logprobs();
        let skip_special_tokens = self.get_skip_special_tokens();
        let formatted_prompt = self.get_formatted_prompt();

        Ok(common::OutputOptions {
            logprobs,
            prompt_logprobs,
            skip_special_tokens,
            formatted_prompt,
        })
    }
}

pub trait DeltaGeneratorExt<ResponseType: Send + 'static + std::fmt::Debug>:
    Send + 'static
{
    fn choice_from_postprocessor(
        &mut self,
        response: common::llm_backend::BackendOutput,
    ) -> Result<ResponseType>;

    /// Gets the current prompt token count (Input Sequence Length).
    fn get_isl(&self) -> Option<u32>;

    /// Creates a final usage-only chunk for OpenAI compliance.
    fn create_usage_chunk(&self) -> ResponseType;

    /// Check if usage tracking is enabled.
    fn is_usage_enabled(&self) -> bool;
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ParsingOptions {
    pub tool_call_parser: Option<String>,

    pub reasoning_parser: Option<String>,
}

impl ParsingOptions {
    pub fn new(tool_call_parser: Option<String>, reasoning_parser: Option<String>) -> Self {
        Self {
            tool_call_parser,
            reasoning_parser,
        }
    }
}
