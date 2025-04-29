// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

const REPLACEMENT_CHAR: char = '_';

/// URL and NATS friendly string.
/// Only a-z, 0-9, - and _.
#[derive(Serialize, Clone, Debug, Eq, PartialEq)]
pub struct Slug(String);

impl Slug {
    fn new(s: String) -> Slug {
        // remove any leading REPLACEMENT_CHAR
        let s = s.trim_start_matches(REPLACEMENT_CHAR).to_string();
        Slug(s)
    }

    /// Create [`Slug`] from a string.
    pub fn from_string(s: impl AsRef<str>) -> Slug {
        Slug::slugify(s.as_ref())
    }

    /// Turn the string into a valid slug, replacing any not-web-or-nats-safe characters with '-'
    pub fn slugify(s: &str) -> Slug {
        let out = s
            .to_lowercase()
            .chars()
            .map(|c| {
                let is_valid = c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_';
                if is_valid {
                    c
                } else {
                    REPLACEMENT_CHAR
                }
            })
            .collect::<String>();
        Slug::new(out)
    }

    /// Like slugify but also add a four byte hash on the end, in case two different strings slug
    /// to the same thing (e.g. because of case differences).
    pub fn slugify_unique(s: &str) -> Slug {
        let out = s
            .to_lowercase()
            .chars()
            .map(|c| {
                let is_valid = c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_';
                if is_valid {
                    c
                } else {
                    REPLACEMENT_CHAR
                }
            })
            .collect::<String>();
        let hash = blake3::hash(s.as_bytes()).to_string();
        let out = format!("{out}_{}", &hash[(hash.len() - 8)..]);
        Slug::new(out)
    }
}

impl fmt::Display for Slug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct InvalidSlugError(char);

impl fmt::Display for InvalidSlugError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Invalid char '{}'. String can only contain a-z, 0-9, - and _.",
            self.0
        )
    }
}

impl std::error::Error for InvalidSlugError {}

impl TryFrom<&str> for Slug {
    type Error = InvalidSlugError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        s.to_string().try_into()
    }
}

impl TryFrom<String> for Slug {
    type Error = InvalidSlugError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        let is_invalid =
            |c: &char| !c.is_ascii_lowercase() && !c.is_ascii_digit() && *c != '-' && *c != '_';
        match s.chars().find(is_invalid) {
            None => Ok(Slug(s)),
            Some(c) => Err(InvalidSlugError(c)),
        }
    }
}

impl<'de> Deserialize<'de> for Slug {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SlugVisitor;

        impl Visitor<'_> for SlugVisitor {
            type Value = Slug;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter
                    .write_str("a valid slug string containing only characters a-z, 0-9, - and _.")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Slug::try_from(v).map_err(de::Error::custom)
            }

            fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Slug::try_from(v.as_ref()).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_string(SlugVisitor)
    }
}

impl AsRef<str> for Slug {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<str> for Slug {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}
