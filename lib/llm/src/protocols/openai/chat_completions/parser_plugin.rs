// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;
use std::path::Path;
use std::ptr::{NonNull, null_mut};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use dynamo_parsers_v2::unified::plugin::{
    ABI_VERSION, ABI_VERSION_V2, Command, CommandV2, Entry, Reply, ReplyV2,
};
use dynamo_parsers_v2::{
    UnifiedParser, UnifiedParserInit, UnifiedParserOutput,
    unified::{
        LoadedGuidance, LoadedPrompt, ToolCallPolicy, UnifiedParserContinuation,
        UnifiedParserInputCapabilities, UnifiedParserRegistration,
    },
};
use libloading::Library;

struct Plugin {
    entry: Entry,
    abi: u32,
    _library: Library,
}

pub(super) fn load(path: &Path) -> Result<UnifiedParserRegistration> {
    ensure!(path.is_absolute(), "parser plugin path must be absolute");
    let digest = || -> Result<blake3::Hash> {
        Ok(blake3::Hasher::new()
            .update_reader(std::fs::File::open(path)?)?
            .finalize())
    };
    let identity = digest()?;
    // SAFETY: an explicitly configured native plugin is trusted process code.
    let library = unsafe { Library::new(path) }.context("open native parser library")?;
    // SAFETY: this versioned symbol has the signature specified by the SDK.
    let (entry, abi) = unsafe {
        match library.get::<Entry>(b"dynamo_parser_plugin_v2\0") {
            Ok(entry) => (*entry, ABI_VERSION_V2),
            Err(_) => (
                *library
                    .get::<Entry>(b"dynamo_parser_plugin_v1\0")
                    .context("library does not export parser plugin ABI v1 or v2")?,
                ABI_VERSION,
            ),
        }
    };
    let plugin = Arc::new(Plugin {
        entry,
        abi,
        _library: library,
    });
    let (mut descriptor, capabilities, continuation, inspect_prompt) = if abi == ABI_VERSION_V2 {
        let (reply, _) = plugin.call_v2(null_mut(), CommandV2::Describe, None)?;
        reply.base.checked()?;
        let manifest = reply.manifest_v2.context("missing v2 parser manifest")?;
        manifest.validate()?;
        let capabilities = manifest.input_capabilities();
        let continuation = manifest.continuation;
        let inspect_prompt = manifest.base.inspect_prompt;
        (
            manifest.descriptor(),
            capabilities,
            continuation,
            inspect_prompt,
        )
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
    ensure!(
        identity == digest()?,
        "parser library changed during startup"
    );
    let features = ParserFeatures {
        preserve: descriptor.preserve_special_tokens,
        input: capabilities,
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
    let inspect_prompt = inspect_prompt.then(|| {
        let plugin = Arc::clone(&plugin);
        let inspect: Arc<LoadedPrompt> = Arc::new(move |request| {
            let (reply, _) = plugin.call(null_mut(), Command::InspectPrompt(request))?;
            reply
                .checked()?
                .prompt
                .context("plugin did not return prompt state")
        });
        inspect
    });
    descriptor.aliases.sort_unstable();
    descriptor.aliases.dedup();
    Ok(UnifiedParserRegistration::loaded(
        descriptor,
        format!("v{abi}:{identity}"),
        guidance,
        inspect_prompt,
        move |tools| {
            let (reply, handle) = plugin.call(null_mut(), Command::Create(tools.to_vec()))?;
            // Acquire ownership before checking a reply so a failed Create cannot leak a handle.
            let parser = NonNull::new(handle).map(|handle| PluginParser {
                plugin: Arc::clone(&plugin),
                handle,
                features,
                ids: Default::default(),
                pending_error: None,
            });
            reply.checked()?;
            Ok(Box::new(
                parser.context("plugin did not create a parser instance")?,
            ))
        },
    ))
}

#[derive(Default)]
struct Response {
    bytes: Option<Vec<u8>>,
    invalid: bool,
}

unsafe extern "C" fn receive(context: *mut c_void, data: *const u8, len: usize) {
    // SAFETY: the synchronous callback receives the response slot supplied in Plugin::call.
    let response = unsafe { &mut *context.cast::<Response>() };
    if response.bytes.is_some() || data.is_null() || len > 64 * 1024 * 1024 {
        response.invalid = true;
    } else {
        // SAFETY: the ABI requires the reply to remain readable for this callback.
        response.bytes = Some(unsafe { std::slice::from_raw_parts(data, len) }.to_vec());
    }
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
        // SAFETY: owned input and response slot outlive the synchronous call.
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
                &response.bytes.context("plugin did not reply")?,
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
        if let Err(error) = reply.and_then(Reply::checked) {
            tracing::error!(%error, "native parser plugin destructor failed");
        }
    }
}

#[derive(Clone, Copy)]
enum HandleKind {
    Parser,
    Checkpoint,
}

trait CheckReply {
    fn checked(self) -> Result<Self>
    where
        Self: Sized;
}
impl CheckReply for Reply {
    fn checked(self) -> Result<Self> {
        if let Some(error) = &self.error {
            anyhow::bail!("{error}");
        }
        Ok(self)
    }
}

struct PluginParser {
    plugin: Arc<Plugin>,
    handle: NonNull<c_void>,
    features: ParserFeatures,
    ids: std::collections::HashMap<usize, String>,
    pending_error: Option<anyhow::Error>,
}

#[derive(Clone, Copy)]
struct ParserFeatures {
    preserve: bool,
    input: UnifiedParserInputCapabilities,
    continuation: bool,
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
    ids: std::collections::HashMap<usize, String>,
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
        self.call(Command::Initialize(init))?.checked()?;
        Ok(())
    }

    fn set_tool_call_policy(&mut self, policy: ToolCallPolicy) -> Result<()> {
        self.call(Command::Policy(policy))?.checked()?;
        Ok(())
    }

    fn preserve_message_phases(&mut self, enabled: bool) {
        if let Err(error) = self
            .call(Command::PreserveMessagePhases(enabled))
            .and_then(Reply::checked)
        {
            self.pending_error = Some(error);
        }
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        ensure!(
            self.features.input.text,
            "plugin does not accept decoded text"
        );
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
            "plugin does not accept token IDs"
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
        match self.call(Command::Reset).and_then(Reply::checked) {
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
