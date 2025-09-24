// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//based on: https://github.com/EricLBuehler/mistral.rs/blob/d970bb5feb863acf8e8ec90de97e18221fb959f1/mistralrs-core/src/pipeline/chat_template.rs

use std::{collections::HashMap, fs::File, path::Path};

use chrono::{DateTime, Local};
use either::Either;
use ggus::{GGufMetaKV, GGufReader};
use memmap2::Mmap;
use minijinja::{Error, ErrorKind, Value, value::Kwargs};
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct AddedTokensDecoder {
    __type: Option<String>,
    pub content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: Option<bool>,
}

pub fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[derive(Debug, Deserialize)]
pub struct BeginEndUnkTok(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

/// Support older tool use patterns where the tool use template was separate from the default/chat template.
/// Modern patterns use a single template with a `tool_use` key, e.g.
///
/// ```jinja
/// {%- if tools is not none and tool_choice is not none %}
/// ```
#[derive(Debug, Deserialize)]
pub struct ChatTemplateValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, Vec<HashMap<String, String>>>,
);

/// If present, pad_token is usually a single value. Deepseek R1 and it's distill's use a map.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct PadTokenValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

#[allow(dead_code)]
#[derive(Debug, Deserialize, Default)]
/// Template for chat models including bos/eos/unk as well as the chat template.
pub struct ChatTemplate {
    pub bos_token: Option<BeginEndUnkTok>,
    pub eos_token: Option<BeginEndUnkTok>,
    pub unk_token: Option<BeginEndUnkTok>,

    /// Jinja format [chat templating] for chat completion.
    ///
    /// [chat templating]: https://huggingface.co/docs/transformers/chat_templating
    pub chat_template: Option<ChatTemplateValue>,

    // future
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    added_tokens_decoder: Option<HashMap<String, AddedTokensDecoder>>,
    additional_special_tokens: Option<Vec<String>>,
    clean_up_tokenization_spaces: Option<bool>,
    device_map: Option<String>,
    legacy: Option<bool>,
    model_max_length: Option<f64>,
    pad_token: Option<PadTokenValue>,
    sp_model_kwargs: Option<HashMap<String, String>>,
    spaces_between_special_tokens: Option<bool>,
    tokenizer_class: Option<String>,
    truncation_size: Option<String>,
    use_default_system_prompt: Option<bool>,
}

impl ChatTemplate {
    pub fn from_gguf(path: &Path) -> anyhow::Result<Self> {
        let file = match File::open(path) {
            Ok(f) => unsafe { Mmap::map(&f)? },
            Err(e) => {
                anyhow::bail!("Failed to open file '{}': {e:?}", path.display());
            }
        };

        let mut reader = GGufReader::new(&file);
        let header = match reader.read_header() {
            Ok(header) => header,
            Err(e) => {
                anyhow::bail!("Failed to read GGUF header of {}: {e:?}", path.display());
            }
        };
        let num_metadata = header.metadata_kv_count;

        let mut out = ChatTemplate::default();

        // bos/eos/unk token conversion
        fn convert(kv: GGufMetaKV) -> Option<BeginEndUnkTok> {
            let id_string: String = kv.read_unsigned().to_string();
            Some(BeginEndUnkTok(Either::Left(id_string)))
        }

        let mut num_found = 0;
        let num_expected = 4; // How many fields we need
        for _ in 0..num_metadata {
            let kv = match reader.read_meta_kv() {
                Ok(kv) => kv,
                Err(err) => anyhow::bail!("read_meta_kv error in '{}': {err:?}", path.display()),
            };
            match kv.key() {
                "tokenizer.ggml.bos_token_id" => {
                    out.bos_token = convert(kv);
                    num_found += 1;
                }
                "tokenizer.ggml.eos_token_id" => {
                    out.eos_token = convert(kv);
                    num_found += 1;
                }
                "tokenizer.ggml.unknown_token_id" => {
                    out.unk_token = convert(kv);
                    num_found += 1;
                }
                "tokenizer.chat_template" => {
                    out.chat_template = kv
                        .value_reader()
                        .read_str()
                        .ok()
                        .map(|s| ChatTemplateValue(Either::Left(s.to_string())));
                    num_found += 1;
                }
                _ => {}
            }
            if num_found == num_expected {
                // No need to look at any more keys
                break;
            }
        }

        Ok(out)
    }

    // pub fn has_chat_template(&self) -> bool {
    //     self.chat_template.is_some()
    // }

    pub fn eos_tok(&self) -> Option<String> {
        match self.eos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn bos_tok(&self) -> Option<String> {
        match self.bos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn unk_tok(&self) -> Option<String> {
        match self.unk_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    #[serde(with = "either::serde_untagged")]
    bos_token_id: Either<u32, Vec<u32>>,
    #[serde(with = "either::serde_untagged")]
    eos_token_id: Either<u32, Vec<u32>>,
}

pub fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    if let Ok(indent) = kwargs.get("indent") {
        let mut buf = Vec::new();
        let repeat = b" ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&repeat);
        let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut serializer).unwrap();
        String::from_utf8(buf).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    } else {
        serde_json::to_string(&value).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    }
    .map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    })
    .map(|s| {
        // When this filter is used the return value is safe for both HTML and JSON
        let mut rv = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '<' => rv.push_str("\\u003c"),
                '>' => rv.push_str("\\u003e"),
                '&' => rv.push_str("\\u0026"),
                '\'' => rv.push_str("\\u0027"),
                _ => rv.push(c),
            }
        }
        Value::from_safe_string(rv)
    })
}

pub fn strftime_now(format_str: &str) -> Result<Value, Error> {
    let local: DateTime<Local> = Local::now();
    Ok(Value::from_safe_string(
        local.format(format_str).to_string(),
    ))
}
