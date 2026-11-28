// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native parser plugins built against the `dynamo-parsers-v2` plugin SDK.
//!
//! A plugin is a Rust `cdylib` that exports `dynamo_parser_plugin_v2` (or `_v1`).
//! Only C ABI values and UTF-8 JSON cross the boundary; the SDK catches panics on
//! the plugin side. Each loaded parser is adapted to a local `dyn UnifiedParser`
//! and admitted to the same registry as the built-in families, so the selectors,
//! descriptor, and guidance it declares follow the built-in routing.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr::{NonNull, null_mut};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Context, Result, ensure};
use dynamo_parsers_v2::unified::plugin::{
    ABI_VERSION, ABI_VERSION_V2, Command, CommandV2, Entry, Reply, ReplyV2,
};
use dynamo_parsers_v2::unified::{
    LoadedGuidance, ToolCallPolicy, UnifiedParserRegistration, register_loaded_unified_parsers,
};
use dynamo_parsers_v2::{
    UnifiedParser, UnifiedParserContinuation, UnifiedParserInit, UnifiedParserInputCapabilities,
    UnifiedParserOutput, unified_parser_registrations,
};
use libloading::Library;

/// Canonical paths already admitted in this process. Loading is add-only, so a
/// repeated flag or a second argument pass in the same process is a no-op.
static LOADED_PATHS: LazyLock<Mutex<HashSet<PathBuf>>> = LazyLock::new(Default::default);

/// Load native parser plugins and admit their registrations together.
///
/// Call at startup, before parser selectors are listed or validated. Returns the
/// canonical families the libraries registered. Conflicting names or aliases,
/// invalid manifests, and capabilities this host cannot serve fail the whole call.
pub fn load_parser_plugins(paths: &[PathBuf]) -> Result<Vec<String>> {
    let mut loaded = LOADED_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    let mut entries = Vec::new();
    let mut canonical_paths = Vec::new();
    for path in paths {
        let canonical = std::fs::canonicalize(path)
            .with_context(|| format!("parser plugin {}", path.display()))?;
        if loaded.contains(&canonical) || canonical_paths.contains(&canonical) {
            continue;
        }
        let entry =
            load(&canonical).with_context(|| format!("parser plugin {}", path.display()))?;
        tracing::info!(
            path = %canonical.display(),
            family = %entry.descriptor.family,
            aliases = ?entry.descriptor.aliases,
            identity = entry.plugin_identity().unwrap_or_default(),
            "Loaded native parser plugin"
        );
        canonical_paths.push(canonical);
        entries.push(entry);
    }
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let families = entries
        .iter()
        .map(|entry| entry.descriptor.family.clone())
        .collect();
    admit(entries)?;
    loaded.extend(canonical_paths);
    Ok(families)
}

/// Register loaded parsers, then drop registry lookups memoized before they existed.
fn admit(entries: Vec<UnifiedParserRegistration>) -> Result<()> {
    reject_legacy_selectors(&entries)?;
    register_loaded_unified_parsers(entries)?;
    super::unified_parser::clear_registry_cache();
    Ok(())
}

/// Auto mode keeps a legacy-only selector such as `phi4` on the legacy parsers, so a
/// plugin that claimed one would take over deployments already configured with it.
fn reject_legacy_selectors(entries: &[UnifiedParserRegistration]) -> Result<()> {
    let legacy: HashSet<&str> = dynamo_parsers::tool_calling::parsers::get_available_tool_parsers()
        .into_iter()
        .chain(dynamo_parsers::reasoning::get_available_reasoning_parsers())
        .collect();
    for entry in entries {
        let descriptor = &entry.descriptor;
        for name in std::iter::once(&descriptor.family)
            .chain(&descriptor.aliases)
            .chain(&descriptor.legacy_configuration_selectors)
        {
            ensure!(
                !legacy.contains(name.as_str()),
                "parser '{}' cannot claim '{name}': it is a legacy tool or reasoning parser selector",
                descriptor.family
            );
        }
    }
    Ok(())
}

/// The effective parser registry, built-ins and plugins, as JSON for operators.
pub fn parser_registry_json() -> serde_json::Value {
    let parsers = unified_parser_registrations()
        .into_iter()
        .map(|entry| {
            let mut value = serde_json::to_value(&entry.descriptor)
                .unwrap_or_else(|error| serde_json::json!({ "error": error.to_string() }));
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "source".into(),
                    if entry.plugin_identity().is_some() {
                        "plugin"
                    } else if entry.is_vendor() {
                        "vendor"
                    } else {
                        "builtin"
                    }
                    .into(),
                );
                object.insert("plugin_identity".into(), entry.plugin_identity().into());
            }
            value
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "abi_versions": [ABI_VERSION, ABI_VERSION_V2],
        "parsers": parsers,
    })
}

fn load(path: &Path) -> Result<UnifiedParserRegistration> {
    let digest = || -> Result<blake3::Hash> {
        Ok(blake3::Hasher::new()
            .update_reader(std::fs::File::open(path)?)?
            .finalize())
    };
    let identity = digest()?;
    // SAFETY: an explicitly configured native plugin is trusted process code.
    let library = unsafe { Library::new(path) }.context("open native parser library")?;
    // SAFETY: these versioned symbols have the signature specified by the SDK.
    let (entry, abi) = unsafe {
        match library.get::<Entry>(b"dynamo_parser_plugin_v2\0") {
            Ok(entry) => (*entry, ABI_VERSION_V2),
            Err(_) => (
                *library
                    .get::<Entry>(b"dynamo_parser_plugin_v1\0")
                    .context("library exports neither parser plugin ABI v1 nor v2")?,
                ABI_VERSION,
            ),
        }
    };
    let registration = registration(entry, abi, format!("v{abi}:{identity}"), Some(library))?;
    ensure!(
        identity == digest()?,
        "parser library changed during startup"
    );
    Ok(registration)
}

/// Build a registration around a plugin entry point. `library` keeps the code
/// mapped for as long as the factory or any parser instance can call it.
fn registration(
    entry: Entry,
    abi: u32,
    identity: String,
    library: Option<Library>,
) -> Result<UnifiedParserRegistration> {
    let plugin = Arc::new(Plugin {
        entry,
        abi,
        _library: library,
    });
    let (mut descriptor, input, continuation, inspect_prompt) = if abi == ABI_VERSION_V2 {
        let (reply, _) = plugin.call_v2(null_mut(), CommandV2::Describe, None)?;
        reply.base.checked()?;
        let manifest = reply.manifest_v2.context("missing v2 parser manifest")?;
        manifest.validate()?;
        let input = manifest.input_capabilities();
        let continuation = manifest.continuation;
        let inspect_prompt = manifest.base.inspect_prompt;
        (manifest.descriptor(), input, continuation, inspect_prompt)
    } else {
        let (reply, _) = plugin.call(null_mut(), Command::Describe)?;
        let manifest = reply
            .checked()?
            .manifest
            .context("missing parser manifest")?;
        let inspect_prompt = manifest.inspect_prompt;
        (
            manifest.descriptor(),
            UnifiedParserInputCapabilities::default(),
            false,
            inspect_prompt,
        )
    };
    check_host_support(&descriptor.family, input, inspect_prompt)?;
    let features = ParserFeatures {
        preserve: descriptor.preserve_special_tokens,
        input,
        continuation,
    };
    let guidance = descriptor.guidance.structural_tag.then(|| {
        let plugin = Arc::clone(&plugin);
        let build: Arc<LoadedGuidance> = Arc::new(move |request| {
            let (reply, _) = plugin.call(null_mut(), Command::Guide(Box::new(request)))?;
            Ok(reply.checked()?.structural_tag)
        });
        build
    });
    descriptor.aliases.sort_unstable();
    descriptor.aliases.dedup();
    Ok(UnifiedParserRegistration::loaded(
        descriptor,
        identity,
        guidance,
        None,
        move |tools| {
            let (reply, handle) = plugin.call(null_mut(), Command::Create(tools.to_vec()))?;
            // Take ownership before checking the reply so a failed Create cannot leak a handle.
            let parser = NonNull::new(handle).map(|handle| PluginParser {
                plugin: Arc::clone(&plugin),
                handle,
                features,
                ids: HashMap::new(),
                pending_error: None,
            });
            reply.checked()?;
            Ok(Box::new(
                parser.context("plugin did not create a parser instance")?,
            ))
        },
    ))
}

/// Reject manifests that ask for host behavior this build does not provide.
fn check_host_support(
    family: &str,
    input: UnifiedParserInputCapabilities,
    inspect_prompt: bool,
) -> Result<()> {
    // Generated output reaches unified parsers as decoded text. Token-ID input is
    // used only to replay the built-in Harmony prompt prefix.
    ensure!(
        input.text,
        "parser '{family}' accepts only token IDs; this Dynamo build passes decoded text to parser plugins"
    );
    // Prompt preprocessing does not call plugin prompt hooks yet, so a parser that
    // relies on one would start in the wrong reasoning state.
    ensure!(
        !inspect_prompt,
        "parser '{family}' inspects the rendered prompt; this Dynamo build does not call prompt hooks of parser plugins"
    );
    Ok(())
}

struct Plugin {
    entry: Entry,
    abi: u32,
    _library: Option<Library>,
}

#[derive(Default)]
struct Response {
    bytes: Option<Vec<u8>>,
    invalid: bool,
}

/// Largest reply the host copies from a plugin.
const MAX_REPLY_BYTES: usize = 64 * 1024 * 1024;

unsafe extern "C" fn receive(context: *mut c_void, data: *const u8, len: usize) {
    // SAFETY: the synchronous callback receives the response slot supplied in Plugin::exchange.
    let response = unsafe { &mut *context.cast::<Response>() };
    if response.bytes.is_some() || data.is_null() || len > MAX_REPLY_BYTES {
        response.invalid = true;
    } else {
        // SAFETY: the ABI requires the reply to stay readable for this callback.
        response.bytes = Some(unsafe { std::slice::from_raw_parts(data, len) }.to_vec());
    }
}

#[derive(Clone, Copy)]
enum HandleKind {
    Parser,
    Checkpoint,
}

impl Plugin {
    fn call(&self, handle: *mut c_void, command: Command) -> Result<(Reply, *mut c_void)> {
        let creates = matches!(command, Command::Create(_)).then_some(HandleKind::Parser);
        if self.abi == ABI_VERSION_V2 {
            let (reply, handle) = self.call_v2(handle, CommandV2::V1(command), creates)?;
            return Ok((reply.base, handle));
        }
        self.exchange(handle, &command, creates)
    }

    fn call_v2(
        &self,
        handle: *mut c_void,
        command: CommandV2,
        creates: Option<HandleKind>,
    ) -> Result<(ReplyV2, *mut c_void)> {
        self.exchange(handle, &command, creates)
    }

    fn exchange<T: serde::de::DeserializeOwned>(
        &self,
        handle: *mut c_void,
        command: &impl serde::Serialize,
        creates: Option<HandleKind>,
    ) -> Result<(T, *mut c_void)> {
        let input = serde_json::to_vec(command)?;
        let mut response = Response::default();
        // SAFETY: the input and response slot outlive the synchronous call.
        let returned = unsafe {
            (self.entry)(
                self.abi,
                handle,
                input.as_ptr(),
                input.len(),
                receive,
                (&mut response as *mut Response).cast(),
            )
        };
        let decoded = (|| -> Result<T> {
            ensure!(!response.invalid, "invalid parser plugin reply");
            ensure!(
                creates.is_some() || returned.is_null(),
                "unexpected parser plugin handle"
            );
            Ok(serde_json::from_slice(
                &response.bytes.context("parser plugin did not reply")?,
            )?)
        })();
        match decoded {
            Ok(reply) => Ok((reply, returned)),
            Err(error) => {
                if let Some(kind) = creates
                    && !returned.is_null()
                {
                    self.destroy(returned, kind);
                }
                Err(error)
            }
        }
    }

    fn destroy(&self, handle: *mut c_void, kind: HandleKind) {
        let reply = match kind {
            HandleKind::Parser => self.call(handle, Command::Destroy).map(|(reply, _)| reply),
            HandleKind::Checkpoint => self
                .call_v2(handle, CommandV2::DropCheckpoint, None)
                .map(|(reply, _)| reply.base),
        };
        if let Err(error) = reply.and_then(CheckReply::checked) {
            tracing::error!(%error, "native parser plugin destructor failed");
        }
    }
}

trait CheckReply: Sized {
    fn checked(self) -> Result<Self>;
}

impl CheckReply for Reply {
    fn checked(self) -> Result<Self> {
        if let Some(error) = &self.error {
            anyhow::bail!("{error}");
        }
        Ok(self)
    }
}

#[derive(Clone, Copy)]
struct ParserFeatures {
    preserve: bool,
    input: UnifiedParserInputCapabilities,
    continuation: bool,
}

struct PluginParser {
    plugin: Arc<Plugin>,
    handle: NonNull<c_void>,
    features: ParserFeatures,
    ids: HashMap<usize, String>,
    /// An error from a trait method that cannot return one, reported on the next call.
    pending_error: Option<anyhow::Error>,
}

// SAFETY: the SDK requires a Send parser and this wrapper exclusively owns the handle.
unsafe impl Send for PluginParser {}

impl Drop for PluginParser {
    fn drop(&mut self) {
        self.plugin
            .destroy(self.handle.as_ptr(), HandleKind::Parser);
    }
}

impl PluginParser {
    fn call(&mut self, command: Command) -> Result<Reply> {
        self.plugin
            .call(self.handle.as_ptr(), command)
            .map(|(reply, _)| reply)
    }

    fn advance(&mut self, command: Command, output: &mut UnifiedParserOutput) -> Result<()> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        let reply = self.call(command)?;
        self.commit(reply, output)
    }

    /// Keep events the plugin committed before it reported an error.
    fn commit(&mut self, mut reply: Reply, output: &mut UnifiedParserOutput) -> Result<()> {
        self.ids.extend(std::mem::take(&mut reply.tool_ids));
        output.events.append(&mut reply.events);
        reply.checked()?;
        Ok(())
    }

    fn finish_with(&mut self, command: Command) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        self.advance(command, &mut output)?;
        Ok(output)
    }
}

struct PluginCheckpoint {
    plugin: Arc<Plugin>,
    handle: NonNull<c_void>,
    features: ParserFeatures,
    ids: HashMap<usize, String>,
}

// SAFETY: this wrapper exclusively owns the Send checkpoint and cannot be shared.
unsafe impl Send for PluginCheckpoint {}

impl Drop for PluginCheckpoint {
    fn drop(&mut self) {
        self.plugin
            .destroy(self.handle.as_ptr(), HandleKind::Checkpoint);
    }
}

impl UnifiedParserContinuation for PluginCheckpoint {
    fn fork(&self) -> Result<Box<dyn UnifiedParser>> {
        let (reply, handle) = self.plugin.call_v2(
            self.handle.as_ptr(),
            CommandV2::Fork,
            Some(HandleKind::Parser),
        )?;
        let parser = NonNull::new(handle).map(|handle| PluginParser {
            plugin: Arc::clone(&self.plugin),
            handle,
            features: self.features,
            ids: self.ids.clone(),
            pending_error: None,
        });
        reply.base.checked()?;
        Ok(Box::new(
            parser.context("plugin did not fork a parser instance")?,
        ))
    }
}

impl UnifiedParser for PluginParser {
    fn continuation(&mut self) -> Result<Option<Box<dyn UnifiedParserContinuation>>> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        if !self.features.continuation {
            return Ok(None);
        }
        let (reply, handle) = self.plugin.call_v2(
            self.handle.as_ptr(),
            CommandV2::Checkpoint,
            Some(HandleKind::Checkpoint),
        )?;
        let checkpoint = NonNull::new(handle).map(|handle| PluginCheckpoint {
            plugin: Arc::clone(&self.plugin),
            handle,
            features: self.features,
            ids: self.ids.clone(),
        });
        reply.base.checked()?;
        Ok(checkpoint.map(|checkpoint| Box::new(checkpoint) as Box<dyn UnifiedParserContinuation>))
    }

    fn input_capabilities(&self) -> UnifiedParserInputCapabilities {
        self.features.input
    }

    fn initialize_request(&mut self, init: UnifiedParserInit) -> Result<()> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        self.call(Command::Initialize(init))?.checked()?;
        Ok(())
    }

    fn set_tool_call_policy(&mut self, policy: ToolCallPolicy) -> Result<()> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        self.call(Command::Policy(policy))?.checked()?;
        Ok(())
    }

    fn preserve_message_phases(&mut self, enabled: bool) {
        if let Err(error) = self
            .call(Command::PreserveMessagePhases(enabled))
            .and_then(CheckReply::checked)
        {
            self.pending_error = Some(error);
        }
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        // `registration` admits only plugins that accept decoded text.
        self.advance(Command::Push(delta.into()), output)
    }

    fn parse_token_ids_into(
        &mut self,
        ids: &[u32],
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        ensure!(
            self.features.input.token_ids,
            "parser plugin does not accept token IDs"
        );
        let (reply, _) = self.plugin.call_v2(
            self.handle.as_ptr(),
            CommandV2::PushTokenIds(ids.to_vec()),
            None,
        )?;
        self.commit(reply.base, output)
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        self.finish_with(Command::Finish)
    }

    fn finish_at_eos(&mut self) -> Result<UnifiedParserOutput> {
        self.finish_with(Command::FinishAtEos)
    }

    fn reset(&mut self) -> String {
        self.ids.clear();
        self.pending_error = None;
        match self.call(Command::Reset).and_then(CheckReply::checked) {
            Ok(reply) => reply.recovered,
            Err(error) => {
                self.pending_error = Some(error);
                String::new()
            }
        }
    }

    fn preserve_special_tokens(&self) -> bool {
        self.features.preserve
    }

    fn tool_call_id(&self, index: usize) -> Option<&str> {
        self.ids.get(&index).map(String::as_str)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! The fixture exports the SDK's v2 entry point from this test binary, so the
    //! adapter is exercised over the real C ABI without building a separate library.

    use dynamo_parsers_v2::unified::plugin::{GuidanceRequest, Hooks, Manifest, ManifestV2};
    use dynamo_parsers_v2::unified::{ForcedToolGuidance, UnifiedParserEvent};
    use dynamo_parsers_v2::{Tool, ToolCallDelta};

    use super::*;

    const FAMILY: &str = "dis3562_abi_fixture";
    pub(crate) const ALIAS: &str = "dis3562-abi-fixture";
    /// A first tool with this name makes the fixture guide build no tag.
    pub(crate) const NO_TAG_TOOL: &str = "no_tag";
    /// A first tool with this name makes the fixture guide fail.
    pub(crate) const FAILING_TAG_TOOL: &str = "failing_tag";

    /// `<<CALL name {json}>>` records; everything else is content.
    #[derive(Clone, Default)]
    struct CallRecords {
        pending: String,
        calls: usize,
    }

    impl UnifiedParser for CallRecords {
        fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
            self.pending.push_str(delta);
            loop {
                match self.pending.find("<<CALL ") {
                    Some(0) => {
                        let Some(end) = self.pending.find(">>") else {
                            return Ok(());
                        };
                        let record = self.pending[7..end].to_owned();
                        self.pending.drain(..end + 2);
                        let (name, arguments) = record
                            .split_once(' ')
                            .context("call record needs a name and arguments")?;
                        output
                            .events
                            .push(UnifiedParserEvent::ToolCall(ToolCallDelta {
                                tool_index: self.calls,
                                name: Some(name.into()),
                                arguments: arguments.into(),
                                complete: true,
                            }));
                        self.calls += 1;
                    }
                    Some(start) => {
                        let text: String = self.pending.drain(..start).collect();
                        output.events.push(UnifiedParserEvent::Text(text));
                    }
                    None => {
                        // Hold back a possible partial opener.
                        let keep = (1..7)
                            .rev()
                            .find(|n| self.pending.ends_with(&"<<CALL "[..*n]))
                            .unwrap_or(0);
                        let split = self.pending.len() - keep;
                        if split > 0 {
                            let text: String = self.pending.drain(..split).collect();
                            output.events.push(UnifiedParserEvent::Text(text));
                        }
                        return Ok(());
                    }
                }
            }
        }

        fn finish(&mut self) -> Result<UnifiedParserOutput> {
            ensure!(
                !self.pending.starts_with("<<CALL"),
                "unfinished call record"
            );
            let mut output = UnifiedParserOutput::default();
            if !self.pending.is_empty() {
                output
                    .events
                    .push(UnifiedParserEvent::Text(std::mem::take(&mut self.pending)));
            }
            Ok(output)
        }

        fn continuation(&mut self) -> Result<Option<Box<dyn UnifiedParserContinuation>>> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    impl UnifiedParserContinuation for CallRecords {
        fn fork(&self) -> Result<Box<dyn UnifiedParser>> {
            Ok(Box::new(self.clone()))
        }
    }

    fn factory(_: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
        Ok(Box::new(CallRecords::default()))
    }

    fn manifest() -> ManifestV2 {
        ManifestV2 {
            base: Manifest {
                family: FAMILY.into(),
                aliases: vec![ALIAS.into()],
                preserve_special_tokens: false,
                native_completion_authoritative: false,
                required_tokens: Vec::new(),
                structural_tag: true,
                inspect_prompt: false,
            },
            text_input: true,
            token_id_input: false,
            continuation: true,
            discard_native_flush_calls_on_truncation: false,
            forced_tool_guidance: ForcedToolGuidance::RequireTagForForced,
        }
    }

    fn guide(request: GuidanceRequest) -> Result<Option<serde_json::Value>> {
        match request.tools.first().map(|tool| tool.name.as_str()) {
            Some(NO_TAG_TOOL) => Ok(None),
            Some(FAILING_TAG_TOOL) => anyhow::bail!("fixture cannot constrain this tool"),
            _ => Ok(Some(serde_json::json!({
                "fixture_tools": request.tools.len(),
                "choice": request.choice,
            }))),
        }
    }

    dynamo_parsers_v2::export_unified_parser_plugin_v2!(
        manifest,
        factory,
        hooks = Hooks {
            guide: Some(guide),
            inspect_prompt: None,
        }
    );

    fn fixture() -> UnifiedParserRegistration {
        registration(
            dynamo_parser_plugin_v2,
            ABI_VERSION_V2,
            "v2:test".into(),
            None,
        )
        .unwrap()
    }

    fn parse(parser: &mut dyn UnifiedParser, chunks: &[&str]) -> Vec<UnifiedParserEvent> {
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            parser.parse_into(chunk, &mut output).unwrap();
        }
        output.events.extend(parser.finish().unwrap().events);
        output.events
    }

    #[test]
    fn plugin_manifest_becomes_a_registration_with_guidance() {
        let entry = fixture();
        assert_eq!(entry.descriptor.family, FAMILY);
        assert_eq!(entry.descriptor.aliases, [ALIAS]);
        assert!(entry.descriptor.text_input && entry.descriptor.continuation);
        assert!(entry.descriptor.guidance.structural_tag);
        assert_eq!(
            entry.descriptor.guidance.forced_tool_choice,
            ForcedToolGuidance::RequireTagForForced
        );
        assert_eq!(entry.plugin_identity(), Some("v2:test"));
        let tag = entry.plugin_guidance().unwrap()(GuidanceRequest {
            tools: vec![Tool {
                name: "lookup".into(),
                description: None,
                parameters: serde_json::json!({"type": "object"}),
                strict: Some(true),
            }],
            choice: dynamo_parsers_v2::unified::plugin::GuidanceChoice::Required,
            parallel_tool_calls: None,
            schema_mode: Default::default(),
            structured_output_schema: None,
            starts_in_reasoning: false,
            options: Default::default(),
        })
        .unwrap();
        assert_eq!(
            tag,
            Some(serde_json::json!({ "fixture_tools": 1, "choice": "required" }))
        );
    }

    #[test]
    fn plugin_parser_streams_and_forks_across_the_abi() {
        let entry = fixture();
        let chunks = ["Hi <<CA", "LL lookup {\"q\":", "1}>> done"];
        let mut parser = entry.create(&[]).unwrap();
        let streamed = parse(parser.as_mut(), &chunks);
        let mut batch = entry.create(&[]).unwrap();
        let whole = parse(batch.as_mut(), &[&chunks.concat()]);
        for events in [&streamed, &whole] {
            let calls: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::ToolCall(call) => {
                        Some((call.name.clone().unwrap(), call.arguments.clone()))
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(calls, [("lookup".to_string(), "{\"q\":1}".to_string())]);
        }

        let mut parent = entry.create(&[]).unwrap();
        let mut sink = UnifiedParserOutput::default();
        parent.parse_into("<<CALL a {}>>", &mut sink).unwrap();
        let checkpoint = parent.continuation().unwrap().unwrap();
        drop(parent);
        let mut fork = checkpoint.fork().unwrap();
        let events = parse(fork.as_mut(), &["<<CALL b {}>>"]);
        assert!(matches!(
            events.as_slice(),
            [UnifiedParserEvent::ToolCall(call)] if call.tool_index == 1
        ));
    }

    #[test]
    fn plugin_errors_cross_the_abi_as_parser_errors() {
        let mut parser = fixture().create(&[]).unwrap();
        let mut output = UnifiedParserOutput::default();
        parser
            .parse_into("<<CALL broken>>", &mut output)
            .unwrap_err();
        let mut parser = fixture().create(&[]).unwrap();
        parser.parse_into("<<CALL", &mut output).unwrap();
        let error = parser.finish().unwrap_err();
        assert!(error.to_string().contains("unfinished call record"));
    }

    /// Admit the fixture to this process's registry. Call it in an isolated test process.
    pub(crate) fn admit_fixture() {
        admit(vec![fixture()]).unwrap();
    }

    #[test]
    fn abi_version_mismatch_is_rejected() {
        // The v2 entry point refuses a v1 request and answers in the v2 reply shape,
        // which carries no v1 manifest; the host never admits the parser.
        let Err(error) = registration(dynamo_parser_plugin_v2, ABI_VERSION, "v1:test".into(), None)
        else {
            panic!("a v1 request to the v2 entry point was admitted");
        };
        assert_eq!(format!("{error:#}"), "missing parser manifest");
    }

    #[test]
    fn plugin_selectors_cannot_claim_built_in_names() {
        let mut entry = fixture();
        entry.descriptor.aliases.push("hermes".into());
        let error = register_loaded_unified_parsers(vec![entry]).unwrap_err();
        assert!(error.to_string().contains("'hermes'"), "{error}");
    }

    #[test]
    fn plugin_selectors_cannot_claim_legacy_only_names() {
        // `phi4` has a legacy tool parser and no unified registration.
        assert!(super::super::unified_parser::canonical_family("phi4").is_none());
        let mut entry = fixture();
        entry.descriptor.aliases.push("phi4".into());
        let error = admit(vec![entry]).unwrap_err();
        assert!(error.to_string().contains("'phi4'"), "{error}");
    }

    #[test]
    fn plugins_that_inspect_the_prompt_are_rejected() {
        let text = UnifiedParserInputCapabilities {
            text: true,
            ..Default::default()
        };
        check_host_support(FAMILY, text, false).unwrap();
        let error = check_host_support(FAMILY, text, true).unwrap_err();
        assert!(error.to_string().contains("prompt hooks"), "{error}");
    }

    #[test]
    fn loaded_plugin_selectors_route_through_unified_selection() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::loaded_plugin_selectors_route_through_unified_selection"
            ),
            &[],
        ) {
            return;
        }
        use super::super::{tool_parser_v2, unified_parser};
        use crate::preprocessor::structural_tag::{StructuralTagDecision, structural_tag_decision};
        use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
        use dynamo_runtime::config::ParserVersion;

        // A miss is memoized until admission clears it.
        assert_eq!(unified_parser::canonical_family(ALIAS), None);
        admit_fixture();
        assert!(tool_parser_v2::unified_family_names().contains(&ALIAS));
        assert_eq!(unified_parser::canonical_family(ALIAS), Some(FAMILY));
        assert_eq!(
            unified_parser::selected_family_for_selection(
                Some(ALIAS),
                Some(FAMILY),
                ParserVersion::Auto
            ),
            Some(FAMILY)
        );
        // Plugins are Unified-only: forced V1 rejects the selector by name.
        let error =
            tool_parser_v2::validate_tool_request_mode(ParserVersion::V1, Some(ALIAS), None, false)
                .unwrap_err();
        assert!(error.to_string().contains(ALIAS), "{error}");

        // A forced choice installs the plugin-built tag without an operator opt-in.
        let tools = [ToolDefinition {
            name: "lookup".into(),
            parameters: Some(serde_json::json!({"type": "object"})),
            strict: None,
        }];
        let decision = structural_tag_decision(
            (Some(ALIAS), None),
            &ToolChoice::Required,
            &tools,
            None,
            None,
            false,
            false,
        )
        .unwrap();
        assert!(matches!(decision, StructuralTagDecision::Required(_)));
    }

    #[test]
    fn missing_library_is_a_startup_error() {
        let error = load_parser_plugins(&["/nonexistent/libparser.so".into()]).unwrap_err();
        assert!(format!("{error:#}").contains("/nonexistent/libparser.so"));
    }
}
