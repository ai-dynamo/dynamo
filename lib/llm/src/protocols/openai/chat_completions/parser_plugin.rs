// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;
use std::path::Path;
use std::ptr::{NonNull, null_mut};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use dynamo_parsers_v2::unified::plugin::{ABI_VERSION, Command, Entry, Reply};
use dynamo_parsers_v2::{
    UnifiedParser, UnifiedParserInit, UnifiedParserOutput,
    unified::{LoadedGuidance, LoadedPrompt, ToolCallPolicy, UnifiedParserRegistration},
};
use libloading::Library;

struct Plugin {
    entry: Entry,
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
    let entry = *unsafe { library.get::<Entry>(b"dynamo_parser_plugin_v1\0") }
        .context("library does not export parser plugin ABI v1")?;
    let plugin = Arc::new(Plugin {
        entry,
        _library: library,
    });
    let (reply, _) = plugin.call(null_mut(), Command::Describe)?;
    let manifest = reply
        .checked()?
        .manifest
        .context("missing parser manifest")?;
    ensure!(
        identity == digest()?,
        "parser library changed during startup"
    );
    let preserve = manifest.preserve_special_tokens;
    let guidance = manifest.structural_tag.then(|| {
        let plugin = Arc::clone(&plugin);
        let build: Arc<LoadedGuidance> = Arc::new(move |request| {
            let (reply, _) = plugin.call(null_mut(), Command::Guide(Box::new(request)))?;
            Ok(reply.checked()?.structural_tag)
        });
        build
    });
    let inspect_prompt = manifest.inspect_prompt.then(|| {
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
    let mut descriptor = manifest.descriptor();
    descriptor.aliases.sort_unstable();
    descriptor.aliases.dedup();
    Ok(UnifiedParserRegistration::loaded(
        descriptor,
        format!("v{ABI_VERSION}:{identity}"),
        guidance,
        inspect_prompt,
        move |tools| {
            let (reply, handle) = plugin.call(null_mut(), Command::Create(tools.to_vec()))?;
            // Acquire ownership before checking a reply so a failed Create cannot leak a handle.
            let parser = NonNull::new(handle).map(|handle| PluginParser {
                plugin: Arc::clone(&plugin),
                handle,
                preserve,
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
        let creates = matches!(command, Command::Create(_));
        let input = serde_json::to_vec(&command)?;
        let mut response = Response::default();
        // SAFETY: owned input and response slot outlive the synchronous call.
        let returned = unsafe {
            (self.entry)(
                ABI_VERSION,
                handle,
                input.as_ptr(),
                input.len(),
                receive,
                (&mut response as *mut Response).cast(),
            )
        };
        let decoded = (|| -> Result<Reply> {
            ensure!(!response.invalid, "invalid parser plugin reply");
            ensure!(
                creates || returned.is_null(),
                "unexpected parser plugin handle"
            );
            Ok(serde_json::from_slice(
                &response.bytes.context("plugin did not reply")?,
            )?)
        })();
        match decoded {
            Ok(reply) => Ok((reply, returned)),
            Err(error) => {
                if creates && !returned.is_null() {
                    self.destroy(returned);
                }
                Err(error)
            }
        }
    }

    fn destroy(&self, handle: *mut c_void) {
        if let Err(error) = self
            .call(handle, Command::Destroy)
            .and_then(|(reply, _)| reply.checked())
        {
            tracing::error!(%error, "native parser plugin destructor failed");
        }
    }
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
    preserve: bool,
    ids: std::collections::HashMap<usize, String>,
    pending_error: Option<anyhow::Error>,
}

// SAFETY: the SDK requires a Send parser and this wrapper exclusively owns the handle.
unsafe impl Send for PluginParser {}

impl Drop for PluginParser {
    fn drop(&mut self) {
        self.plugin.destroy(self.handle.as_ptr());
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
        let mut reply = self.call(command)?;
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

impl UnifiedParser for PluginParser {
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
        self.advance(Command::Push(delta.into()), output)
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
        self.preserve
    }

    fn tool_call_id(&self, index: usize) -> Option<&str> {
        self.ids.get(&index).map(String::as_str)
    }
}
