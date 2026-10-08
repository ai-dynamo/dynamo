// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::io::{self, Write};

use serde::Serialize;
#[cfg(target_os = "linux")]
#[path = "support/atomic_file.rs"]
mod atomic_file;
#[cfg(target_os = "linux")]
#[path = "support/secure_file.rs"]
mod secure_file;

const USAGE: &str = concat!(
    "Usage: model-protection-enroll inspect [--device /dev/tpmrmN]\n",
    "       model-protection-enroll provision --state PATH --journal PATH --policy-public PATH --ek-handle 0xXXXXXXXX --ak-handle 0xXXXXXXXX --duk-handle 0xXXXXXXXX --confirm-new-handles true [--device /dev/tpmrmN]\n",
    "       model-protection-enroll request --state PATH --binding PATH --ek-certificate-chain PATH --output PATH [--device /dev/tpmrmN]\n",
    "       model-protection-enroll respond --state PATH --request PATH --challenge PATH --challenge-signature PATH --challenge-key-id ID --challenge-public-key PATH --output PATH [--device /dev/tpmrmN]\n",
    "Experimental; not hardware-qualified. Only provision creates persistent objects; it never overwrites occupied targets. Keep its private journal on the customer host. respond requires v2 state with the original DUK creation ticket and a signed, unexpired authority challenge. No certificate issuance."
);

#[derive(Serialize)]
struct Failure {
    format: &'static str,
    format_version: u16,
    code: &'static str,
    production_ready: bool,
}

fn parse_args(args: &[String]) -> Result<&str, &'static str> {
    let device = match args {
        [command] if command == "inspect" => "/dev/tpmrm0",
        [command, option, device] if command == "inspect" && option == "--device" => device,
        _ => return Err("ENROLLMENT_CONFIG_INVALID"),
    };
    let number = device
        .strip_prefix("/dev/tpmrm")
        .filter(|value| {
            !value.is_empty() && value.len() <= 3 && value.bytes().all(|byte| byte.is_ascii_digit())
        })
        .ok_or("ENROLLMENT_CONFIG_INVALID")?;
    let _: u8 = number.parse().map_err(|_| "ENROLLMENT_CONFIG_INVALID")?;
    Ok(device)
}

fn write_json(value: &impl Serialize) -> Result<(), &'static str> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, value).map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
    stdout
        .write_all(b"\n")
        .map_err(|_| "ENROLLMENT_OUTPUT_FAILED")
}

fn run(args: &[String]) -> Result<(), &'static str> {
    #[cfg(target_os = "linux")]
    if args.first().is_some_and(|command| command == "request") {
        return linux::request(args);
    }
    #[cfg(target_os = "linux")]
    if args.first().is_some_and(|command| command == "respond") {
        return linux::respond(args);
    }
    #[cfg(target_os = "linux")]
    if args.first().is_some_and(|command| command == "provision") {
        return linux::provision(args);
    }
    let device = parse_args(args)?;
    #[cfg(target_os = "linux")]
    {
        write_json(&linux::inspect(device)?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = device;
        Err("ENROLLMENT_PLATFORM_UNSUPPORTED")
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("{USAGE}");
        return;
    }
    if let Err(code) = run(&args) {
        let failure = Failure {
            format: "model-protection-enrollment-error",
            format_version: 1,
            code,
            production_ready: false,
        };
        if write_json(&failure).is_err() {
            eprintln!("model_protection event=tpm_inspect_failed code=ENROLLMENT_OUTPUT_FAILED");
        }
        std::process::exit(1);
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs;
    use std::os::unix::fs::{FileTypeExt, MetadataExt};
    use std::path::Path;
    use std::str::FromStr;

    use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
    use dynamo_model_protection::enrollment::format::{
        Binding, MAX_BUNDLE_BYTES, Request, hex, parse, validate_request,
    };
    use serde::Serialize;
    use tss_esapi::constants::{CapabilityType, PropertyTag};
    use tss_esapi::handles::{KeyHandle, PersistentTpmHandle, TpmHandle};
    use tss_esapi::structures::CapabilityData;
    use tss_esapi::traits::Marshall;
    use tss_esapi::{Context, TctiNameConf};

    #[derive(Serialize)]
    pub(super) struct Inventory {
        format: &'static str,
        format_version: u16,
        device: String,
        device_gid: u32,
        family: String,
        manufacturer: String,
        revision: u32,
        firmware_version_1: String,
        firmware_version_2: String,
        persistent_handles: Handles,
        nv_indices: Handles,
        ek_certificate_verification: &'static str,
        production_ready: bool,
    }

    #[derive(Serialize)]
    struct Handles {
        values: Vec<String>,
        truncated: bool,
    }

    fn device_gid(device: &Path) -> Result<u32, &'static str> {
        let metadata = fs::symlink_metadata(device).map_err(|_| "TPM_DEVICE_UNAVAILABLE")?;
        if !metadata.file_type().is_char_device() {
            return Err("TPM_DEVICE_INVALID");
        }
        Ok(metadata.gid())
    }

    fn property(context: &mut Context, tag: PropertyTag) -> Result<u32, &'static str> {
        context
            .get_tpm_property(tag)
            .map_err(|_| "TPM_INSPECTION_FAILED")?
            .ok_or("TPM_PROPERTY_UNAVAILABLE")
    }

    fn handles(context: &mut Context, first: u32) -> Result<Handles, &'static str> {
        let (capability, truncated) = context
            .get_capability(CapabilityType::Handles, first, 64)
            .map_err(|_| "TPM_INSPECTION_FAILED")?;
        let CapabilityData::Handles(handles) = capability else {
            return Err("TPM_INSPECTION_FAILED");
        };
        let values = handles
            .into_inner()
            .into_iter()
            .map(|handle| format!("0x{:08x}", u32::from(handle)))
            .collect();
        Ok(Handles { values, truncated })
    }

    pub(super) fn inspect(device: &str) -> Result<Inventory, &'static str> {
        let device_gid = device_gid(Path::new(device))?;
        let tcti = TctiNameConf::from_str(&format!("device:{device}"))
            .map_err(|_| "ENROLLMENT_CONFIG_INVALID")?;
        // Never use environment-selected TCTIs or simulator fallback for inventory.
        let mut context = Context::new(tcti).map_err(|_| "TPM_UNAVAILABLE")?;
        let family = property(&mut context, PropertyTag::FamilyIndicator)?;
        if family != 0x322e_3000 {
            return Err("TPM_VERSION_UNSUPPORTED");
        }
        Ok(Inventory {
            format: "model-protection-tpm-inventory",
            format_version: 1,
            device: device.to_owned(),
            device_gid,
            family: format!("0x{family:08x}"),
            manufacturer: format!(
                "0x{:08x}",
                property(&mut context, PropertyTag::Manufacturer)?
            ),
            revision: property(&mut context, PropertyTag::Revision)?,
            firmware_version_1: format!(
                "0x{:08x}",
                property(&mut context, PropertyTag::FirmwareVersion1)?
            ),
            firmware_version_2: format!(
                "0x{:08x}",
                property(&mut context, PropertyTag::FirmwareVersion2)?
            ),
            persistent_handles: handles(&mut context, 0x8100_0000)?,
            nv_indices: handles(&mut context, 0x0100_0000)?,
            ek_certificate_verification: "not_performed",
            production_ready: false,
        })
    }

    type ActivationState = dynamo_model_protection::enrollment::provision::ActivationState;

    pub(super) fn provision(args: &[String]) -> Result<(), &'static str> {
        use dynamo_model_protection::enrollment::provision;
        use std::collections::BTreeMap;
        let mut parsed = BTreeMap::new();
        let mut pairs = args[1..].chunks_exact(2);
        for pair in &mut pairs {
            if !matches!(
                pair[0].as_str(),
                "--state"
                    | "--journal"
                    | "--policy-public"
                    | "--ek-handle"
                    | "--ak-handle"
                    | "--duk-handle"
                    | "--confirm-new-handles"
                    | "--device"
            ) || pair[1].is_empty()
                || parsed.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err("ENROLLMENT_CONFIG_INVALID");
            }
        }
        if !pairs.remainder().is_empty() {
            return Err("ENROLLMENT_CONFIG_INVALID");
        }
        let required = |name| parsed.get(name).copied().ok_or("ENROLLMENT_CONFIG_INVALID");
        if required("--confirm-new-handles")? != "true" {
            return Err("ENROLLMENT_PROVISION_CONFIRMATION_REQUIRED");
        }
        let handle = |name| {
            let value = required(name)?;
            if value.len() != 10 || !value.starts_with("0x") {
                return Err("ENROLLMENT_CONFIG_INVALID");
            }
            u32::from_str_radix(&value[2..], 16).map_err(|_| "ENROLLMENT_CONFIG_INVALID")
        };
        let targets = provision::Handles {
            ek: handle("--ek-handle")?,
            ak: handle("--ak-handle")?,
            duk: handle("--duk-handle")?,
        };
        targets
            .validate()
            .map_err(|_| "ENROLLMENT_CONFIG_INVALID")?;
        let state_path = Path::new(required("--state")?);
        let journal_path = Path::new(required("--journal")?);
        if !state_path.is_absolute() || !journal_path.is_absolute() || state_path == journal_path {
            return Err("ENROLLMENT_CONFIG_INVALID");
        }
        let policy =
            super::secure_file::read_bounded(Path::new(required("--policy-public")?), 88, false)
                .map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        let device = parsed.get("--device").copied().unwrap_or("/dev/tpmrm0");
        super::parse_args(&["inspect".into(), "--device".into(), device.into()])?;
        device_gid(Path::new(device))?;
        let mut context = Context::new(
            TctiNameConf::from_str(&format!("device:{device}"))
                .map_err(|_| "ENROLLMENT_CONFIG_INVALID")?,
        )
        .map_err(|_| "TPM_UNAVAILABLE")?;
        let prepared: provision::PreparedEnrollment = match fs::symlink_metadata(journal_path) {
            Ok(_) => parse(
                &super::secure_file::read_bounded(journal_path, MAX_BUNDLE_BYTES as u64, true)
                    .map_err(|_| "ENROLLMENT_INPUT_INVALID")?,
            )
            .map_err(|_| "ENROLLMENT_INPUT_INVALID")?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if fs::symlink_metadata(state_path).is_ok() {
                    return Err("ENROLLMENT_STATE_ALREADY_EXISTS");
                }
                let prepared = provision::prepare(&mut context, targets, &policy)
                    .map_err(|_| "ENROLLMENT_PREPARE_FAILED_OR_HANDLE_OCCUPIED")?;
                let bytes =
                    serde_json::to_vec(&prepared).map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
                // Commit is forbidden until the private journal has been durably published.
                super::atomic_file::publish_new(journal_path, &bytes)
                    .map_err(|_| "ENROLLMENT_JOURNAL_OUTPUT_FAILED")?;
                prepared
            }
            Err(_) => return Err("ENROLLMENT_INPUT_INVALID"),
        };
        prepared
            .validate_inputs(&targets, &policy)
            .map_err(|_| "ENROLLMENT_JOURNAL_BINDING_INVALID")?;
        let state_bytes =
            serde_json::to_vec(prepared.state()).map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
        // Refuse a different state file before any persistent mutation on resume.
        if fs::symlink_metadata(state_path).is_ok()
            && super::secure_file::read_bounded(state_path, MAX_BUNDLE_BYTES as u64, true)
                .map_err(|_| "ENROLLMENT_INPUT_INVALID")?
                != state_bytes
        {
            return Err("ENROLLMENT_STATE_BINDING_INVALID");
        }
        provision::commit(&mut context, &prepared)
            .map_err(|_| "ENROLLMENT_COMMIT_FAILED_OR_HANDLE_OCCUPIED")?;
        match super::atomic_file::publish_new(state_path, &state_bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if super::secure_file::read_bounded(state_path, MAX_BUNDLE_BYTES as u64, true)
                    .map_err(|_| "ENROLLMENT_INPUT_INVALID")?
                    != state_bytes
                {
                    return Err("ENROLLMENT_STATE_BINDING_INVALID");
                }
            }
            Err(_) => return Err("ENROLLMENT_STATE_OUTPUT_FAILED_RETAIN_JOURNAL"),
        }
        super::write_json(
            &serde_json::json!({"format":"model-protection-provision-result","format_version":1,"duk_name":prepared.state().duk_name,"policy_authority_name":prepared.state().policy_authority_name,"production_ready":false}),
        )
    }

    pub(super) fn respond(args: &[String]) -> Result<(), &'static str> {
        use dynamo_model_protection::enrollment::{
            client,
            format::{identifier, verify_challenge},
        };
        use std::collections::BTreeMap;
        let mut parsed = BTreeMap::new();
        let mut pairs = args[1..].chunks_exact(2);
        for pair in &mut pairs {
            if !matches!(
                pair[0].as_str(),
                "--state"
                    | "--request"
                    | "--challenge"
                    | "--challenge-signature"
                    | "--challenge-key-id"
                    | "--challenge-public-key"
                    | "--output"
                    | "--device"
            ) || pair[1].is_empty()
                || parsed.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err("ENROLLMENT_CONFIG_INVALID");
            }
        }
        if !pairs.remainder().is_empty() {
            return Err("ENROLLMENT_CONFIG_INVALID");
        }
        let required = |name| parsed.get(name).copied().ok_or("ENROLLMENT_CONFIG_INVALID");
        let read = |name, private| {
            super::secure_file::read_bounded(
                Path::new(required(name)?),
                MAX_BUNDLE_BYTES as u64,
                private,
            )
            .map_err(|_| "ENROLLMENT_INPUT_INVALID")
        };
        let device = parsed.get("--device").copied().unwrap_or("/dev/tpmrm0");
        super::parse_args(&["inspect".into(), "--device".into(), device.into()])?;
        let state: ActivationState =
            parse(&read("--state", true)?).map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        state.validate().map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        let ticket = state
            .duk_creation_ticket
            .as_ref()
            .ok_or("ENROLLMENT_CREATION_TICKET_REQUIRED")?;
        let ticket_bytes = BASE64
            .decode(ticket)
            .map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        if ticket_bytes.len() > 72 || BASE64.encode(&ticket_bytes) != *ticket {
            return Err("ENROLLMENT_INPUT_INVALID");
        }
        let request = validate_request(
            parse(&read("--request", false)?).map_err(|_| "ENROLLMENT_INPUT_INVALID")?,
        )
        .map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        if state.duk_name != hex(request.duk_name())
            || state.ak_name != hex(request.ak_name())
            || state.duk_creation_hash != request.request().duk_creation_hash
            || state.policy_authority_name != request.request().binding.policy_authority_name
        {
            return Err("ENROLLMENT_BINDING_INVALID");
        }
        let key_id = required("--challenge-key-id")?;
        if !identifier(key_id) {
            return Err("ENROLLMENT_CONFIG_INVALID");
        }
        let key: [u8; 32] = read("--challenge-public-key", false)?
            .try_into()
            .map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| "ENROLLMENT_CONFIG_INVALID")?
            .as_secs();
        let challenge = verify_challenge(
            &read("--challenge", false)?,
            &read("--challenge-signature", false)?,
            key_id,
            &key,
            &request,
            now,
        )
        .map_err(|_| "ENROLLMENT_CHALLENGE_INVALID")?;
        // Reject unsigned/mixed/expired inputs before opening a TPM connection.
        device_gid(Path::new(device))?;
        let mut context = Context::new(
            TctiNameConf::from_str(&format!("device:{device}"))
                .map_err(|_| "ENROLLMENT_CONFIG_INVALID")?,
        )
        .map_err(|_| "TPM_UNAVAILABLE")?;
        let bytes = client::respond(
            &mut context,
            device,
            &client::ActivationHandles {
                ek: state.ek_handle,
                ak: state.ak_handle,
                duk: state.duk_handle,
            },
            &request,
            &challenge,
            &ticket_bytes,
            now,
        )
        .map_err(|_| "ENROLLMENT_ACTIVATION_OR_CREATION_PROOF_FAILED")?;
        super::atomic_file::publish_new(Path::new(required("--output")?), &bytes)
            .map_err(|_| "ENROLLMENT_OUTPUT_EXISTS_OR_INVALID")?;
        super::write_json(
            &serde_json::json!({"format":"model-protection-enrollment-response-result","format_version":1,"challenge_id":challenge.challenge_id,"production_ready":false}),
        )
    }

    pub(super) fn request(args: &[String]) -> Result<(), &'static str> {
        use std::collections::BTreeMap;
        let mut parsed = BTreeMap::new();
        let mut pairs = args[1..].chunks_exact(2);
        for pair in &mut pairs {
            if !matches!(
                pair[0].as_str(),
                "--state" | "--binding" | "--ek-certificate-chain" | "--output" | "--device"
            ) || pair[1].is_empty()
                || parsed.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err("ENROLLMENT_CONFIG_INVALID");
            }
        }
        if !pairs.remainder().is_empty() {
            return Err("ENROLLMENT_CONFIG_INVALID");
        }
        let required = |name| parsed.get(name).copied().ok_or("ENROLLMENT_CONFIG_INVALID");
        let device = parsed.get("--device").copied().unwrap_or("/dev/tpmrm0");
        super::parse_args(&["inspect".into(), "--device".into(), device.into()])?;
        device_gid(Path::new(device))?;
        let read = |name, private| {
            super::secure_file::read_bounded(
                Path::new(required(name)?),
                MAX_BUNDLE_BYTES as u64,
                private,
            )
            .map_err(|_| "ENROLLMENT_INPUT_INVALID")
        };
        let state: ActivationState =
            parse(&read("--state", true)?).map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        state.validate().map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        let handles = [state.ek_handle, state.ak_handle, state.duk_handle];
        if handles
            .iter()
            .enumerate()
            .any(|(i, h)| handles[..i].contains(h))
        {
            return Err("ENROLLMENT_INPUT_INVALID");
        }
        let binding: Binding =
            parse(&read("--binding", false)?).map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        if binding.policy_authority_name != state.policy_authority_name {
            return Err("ENROLLMENT_BINDING_INVALID");
        }
        let certificates: Vec<String> = parse(&read("--ek-certificate-chain", false)?)
            .map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        let mut context = Context::new(
            TctiNameConf::from_str(&format!("device:{device}"))
                .map_err(|_| "ENROLLMENT_CONFIG_INVALID")?,
        )
        .map_err(|_| "TPM_UNAVAILABLE")?;
        let mut publics = Vec::new();
        let mut qualified_names = Vec::new();
        for (handle, expected_name) in
            handles
                .into_iter()
                .zip([&state.ek_name, &state.ak_name, &state.duk_name])
        {
            let handle =
                PersistentTpmHandle::new(handle).map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
            let key = context
                .tr_from_tpm_public(TpmHandle::Persistent(handle))
                .map(KeyHandle::from)
                .map_err(|_| "TPM_HANDLE_UNAVAILABLE")?;
            let (public, name, qualified) = context
                .read_public(key)
                .map_err(|_| "TPM_INSPECTION_FAILED")?;
            if hex(name.value()) != *expected_name {
                return Err("TPM_HANDLE_BINDING_INVALID");
            }
            publics.push(public.marshall().map_err(|_| "TPM_INSPECTION_FAILED")?);
            qualified_names.push(hex(qualified.value()));
        }
        if let Some(handle) = state.policy_authority_handle {
            let key = context
                .tr_from_tpm_public(TpmHandle::Persistent(
                    PersistentTpmHandle::new(handle).map_err(|_| "ENROLLMENT_INPUT_INVALID")?,
                ))
                .map(KeyHandle::from)
                .map_err(|_| "TPM_HANDLE_UNAVAILABLE")?;
            if hex(context
                .read_public(key)
                .map_err(|_| "TPM_INSPECTION_FAILED")?
                .1
                .value())
                != state.policy_authority_name
            {
                return Err("TPM_HANDLE_BINDING_INVALID");
            }
        }
        let request = Request {
            format: "model-protection-enrollment-request".into(),
            format_version: 1,
            request_id: uuid::Uuid::new_v4().simple().to_string(),
            binding,
            ek_public: BASE64.encode(&publics[0]),
            ek_certificate_chain: certificates,
            ak_public: BASE64.encode(&publics[1]),
            ak_qualified_name: qualified_names[1].clone(),
            duk_public: BASE64.encode(&publics[2]),
            duk_creation_hash: state.duk_creation_hash,
        };
        let validated = validate_request(request).map_err(|_| "ENROLLMENT_INPUT_INVALID")?;
        let bytes =
            serde_json::to_vec(validated.request()).map_err(|_| "ENROLLMENT_OUTPUT_FAILED")?;
        let output = Path::new(required("--output")?);
        super::atomic_file::publish_new(output, &bytes)
            .map_err(|_| "ENROLLMENT_OUTPUT_EXISTS_OR_INVALID")?;
        super::write_json(
            &serde_json::json!({"format":"model-protection-enrollment-request-result","format_version":1,
            "request_sha256":hex(validated.digest()),"production_ready":false}),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rejects_regular_files_and_symlinks_without_tpm_io() {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("device");
            fs::write(&path, b"not a TPM").unwrap();
            assert_eq!(device_gid(&path), Err("TPM_DEVICE_INVALID"));
            let link = directory.path().join("link");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            assert_eq!(device_gid(&link), Err("TPM_DEVICE_INVALID"));
            assert_eq!(
                device_gid(&directory.path().join("missing")),
                Err("TPM_DEVICE_UNAVAILABLE")
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_accepts_read_only_inventory_on_explicit_device_tcti() {
        assert_eq!(parse_args(&["inspect".into()]), Ok("/dev/tpmrm0"));
        assert_eq!(
            parse_args(&["inspect".into(), "--device".into(), "/dev/tpmrm1".into()]),
            Ok("/dev/tpmrm1")
        );
        for args in [
            vec!["provision"],
            vec!["inspect", "--device", "swtpm:host=localhost,port=2321"],
            vec!["inspect", "--device", "/dev/tpmrm0/../secret"],
            vec!["inspect", "--device", "/dev/tpmrm256"],
            vec!["inspect", "--device", "/dev/tpmrm0", "--force"],
        ] {
            let args: Vec<String> = args.into_iter().map(String::from).collect();
            assert_eq!(parse_args(&args), Err("ENROLLMENT_CONFIG_INVALID"));
        }
    }
}
