// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap, HashSet};

use dynamo_protocols::types::responses::{
    InputItem, InputParam, Item, NamespaceToolParamTool, Tool,
};

use super::ResponsesConversionError;

/// Reversible, request-local names for function tools sent to Chat Completions.
/// Unique bare names remain unchanged. Collisions receive short aliases that
/// cannot shadow any original name, including names in conversation history.
#[derive(Clone, Debug, Default)]
pub struct ToolNameMap {
    by_identity: HashMap<(Option<String>, String), String>,
    by_alias: HashMap<String, (Option<String>, String)>,
    available: BTreeSet<(Option<String>, String)>,
}

impl ToolNameMap {
    /// Build deterministic aliases from current tools and historical calls.
    /// Historical identities reserve names but are not available for tool selection.
    pub fn new(tools: &[Tool], input: Option<&InputParam>) -> Self {
        let mut identities = BTreeSet::new();
        for tool in tools {
            match tool {
                Tool::Function(function) => {
                    identities.insert((None, function.name.clone()));
                }
                Tool::Namespace(namespace) => {
                    for tool in &namespace.tools {
                        if let NamespaceToolParamTool::Function(function) = tool {
                            identities
                                .insert((Some(namespace.name.clone()), function.name.clone()));
                        }
                    }
                }
                _ => {}
            }
        }
        let available = identities.clone();
        if let Some(InputParam::Items(items)) = input {
            for item in items {
                if let InputItem::Item(Item::FunctionCall(call)) = item {
                    identities.insert((call.namespace.clone(), call.name.clone()));
                }
            }
        }
        let mut counts = HashMap::new();
        for (_, name) in &identities {
            *counts.entry(name.as_str()).or_insert(0) += 1;
        }
        let mut reserved: HashSet<String> = counts.keys().map(|name| (*name).to_owned()).collect();
        let mut result = Self {
            available,
            ..Self::default()
        };
        let mut next_alias = 0;
        for identity in &identities {
            let name = &identity.1;
            let alias = if counts[name.as_str()] == 1 {
                name.clone()
            } else {
                // Keep the function's meaning visible to the model, using only
                // Chat Completions name characters and room for a unique suffix.
                let qualified = format!("{}__{name}", identity.0.as_deref().unwrap_or("function"));
                let stem: String = qualified
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .take(48)
                    .collect();
                let mut candidate = stem.clone();
                while !reserved.insert(candidate.clone()) {
                    candidate = format!("{stem}_{next_alias}");
                    next_alias += 1;
                }
                candidate
            };
            result.by_identity.insert(identity.clone(), alias.clone());
            result.by_alias.insert(alias, identity.clone());
        }
        result
    }

    /// Return the backend alias, preserving the bare name for unknown identities.
    pub(super) fn encode(&self, namespace: Option<&str>, name: &str) -> String {
        self.by_identity
            .get(&(namespace.map(str::to_owned), name.to_owned()))
            .cloned()
            .unwrap_or_else(|| name.to_owned())
    }

    /// Restore the original namespace and name, treating unknown aliases as bare names.
    pub(super) fn decode<'a>(&'a self, alias: &'a str) -> (Option<&'a str>, &'a str) {
        self.by_alias
            .get(alias)
            .map(|(namespace, name)| (namespace.as_deref(), name.as_str()))
            .unwrap_or((None, alias))
    }

    /// Older clients omit namespaces in choices. Resolve a top-level function
    /// first, or a unique namespaced function; never guess between namespaces.
    pub(super) fn resolve_choice(
        &self,
        namespace: Option<&str>,
        name: &str,
    ) -> anyhow::Result<String> {
        let identity = (namespace.map(str::to_owned), name.to_owned());
        if self.available.contains(&identity) {
            return Ok(self.encode(namespace, name));
        }
        if namespace.is_none() {
            let mut matches = self
                .available
                .iter()
                .filter(|(_, candidate)| candidate == name);
            if let Some((namespace, name)) = matches.next() {
                if matches.next().is_none() {
                    return Ok(self.encode(namespace.as_deref(), name));
                }
                return Err(ResponsesConversionError::InvalidArgument(format!(
                    "Responses tool_choice function '{name}' is ambiguous without a namespace"
                ))
                .into());
            }
        }
        // Preserve legacy unknown-name validation downstream, but never let an
        // unknown client name accidentally select a generated backend alias.
        if namespace.is_none() && !self.by_alias.contains_key(name) {
            return Ok(name.to_owned());
        }
        Err(ResponsesConversionError::InvalidArgument(format!(
            "Responses tool_choice references unknown function '{name}'"
        ))
        .into())
    }
}
