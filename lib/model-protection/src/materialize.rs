// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::OwnedFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};

use rustix::fs::{
    AtFlags, CWD, FileType, FlockOperation, Mode, OFlags, RawDir, RenameFlags, ResolveFlags,
    StatxAttributes, StatxFlags, flock, fstat, fstatfs, fstatvfs, mkdirat, openat2, renameat_with,
    statat, statx, unlinkat,
};
use rustix::io::fcntl_dupfd_cloexec;
use rustix::process::geteuid;
use sha2::{Digest, Sha256};

use crate::format::{
    MAX_MANIFEST_BYTES, ProtectedFile, PublicFile, VerifiedManifest, decode_lower_hex,
    validate_relative_path, validate_safetensors_index,
};
use crate::records::decrypt_records;
use crate::{AuthorizedModel, CancellationToken, ProtectionError, Result, SecretDek};

const TMPFS_MAGIC: i64 = 0x0102_1994;
const COPY_BUFFER_BYTES: usize = 64 * 1024;
const MAX_NAMESPACE_BYTES: usize = 128;
const MAX_CLEANUP_ENTRIES: usize = 8192;
const MAX_CLEANUP_DEPTH: usize = 32;
const SAFE_RESOLVE: ResolveFlags = ResolveFlags::BENEATH
    .union(ResolveFlags::NO_SYMLINKS)
    .union(ResolveFlags::NO_MAGICLINKS)
    .union(ResolveFlags::NO_XDEV);

pub fn validate_namespace(namespace: &str) -> Result<()> {
    let bytes = namespace.as_bytes();
    if bytes.is_empty()
        || bytes.len() > MAX_NAMESPACE_BYTES
        || !bytes[0].is_ascii_alphanumeric()
        || !bytes[bytes.len() - 1].is_ascii_alphanumeric()
        || bytes
            .iter()
            .any(|byte| !byte.is_ascii_alphanumeric() && !matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(ProtectionError::InvalidPackage("namespace"));
    }
    Ok(())
}

pub fn secure_model_root(namespace: &str) -> Result<PathBuf> {
    validate_namespace(namespace)?;
    Ok(PathBuf::from(format!("/run/{namespace}-models")))
}

pub fn load_verified_manifest(
    package_root: &Path,
    expected_key_id: &str,
    verifying_key: &[u8; 32],
) -> Result<VerifiedManifest> {
    let package = SafeDir::open(package_root, false)?;
    let manifest = package.read_bounded("model.protection.json", MAX_MANIFEST_BYTES)?;
    let signature =
        package.read_bounded("model.protection.sig", crate::format::MAX_SIGNATURE_BYTES)?;
    let verified = crate::verify_manifest(&manifest, &signature, expected_key_id, verifying_key)?;
    crate::enforce_runtime_version(&verified, env!("CARGO_PKG_VERSION"))?;
    Ok(verified)
}

pub fn load_authorized_model(
    package_root: &Path,
    license_root: &Path,
    package_key_id: &str,
    package_verifying_key: &[u8; 32],
    license_key_id: &str,
    license_verifying_key: &[u8; 32],
) -> Result<AuthorizedModel> {
    if package_key_id == license_key_id || package_verifying_key == license_verifying_key {
        return Err(ProtectionError::LicenseInvalid("trust domain"));
    }
    let manifest = load_verified_manifest(package_root, package_key_id, package_verifying_key)?;
    let license_root = SafeDir::open(license_root, false)?;
    let license = license_root.read_bounded(
        "model.protection.license.json",
        crate::license::MAX_LICENSE_BYTES,
    )?;
    let signature = license_root.read_bounded(
        "model.protection.license.sig",
        crate::format::MAX_SIGNATURE_BYTES,
    )?;
    crate::license::verify_license(
        &license,
        &signature,
        license_key_id,
        license_verifying_key,
        manifest,
    )
}

/// Materialize one already-verified manifest into an empty private tmpfs model directory.
///
/// The caller owns the surrounding session and must remove it on any returned error. This function
/// removes every declared output it may have created before returning an error.
#[cfg(test)]
fn materialize_model(
    package_root: &Path,
    model: &SafeDir,
    verified: &VerifiedManifest,
    key: SecretDek,
) -> Result<()> {
    let manifest = verified.manifest();
    manifest.validate()?;
    let package = SafeDir::open(package_root, false)?;
    model.require_empty()?;
    let artifact_id = manifest.artifact_id_bytes()?;
    let nonce_prefix = manifest.nonce_prefix_bytes()?;

    let mut created_outputs = Vec::new();
    let result = (|| {
        let mut safetensors_index = None;
        for file in &manifest.public_files {
            if let Some(bytes) = copy_public_file(&package, model, file)? {
                safetensors_index = Some(bytes);
            }
            created_outputs.push(file.output_path.as_str());
        }
        validate_safetensors_index(manifest, safetensors_index.as_deref())?;
        for file in &manifest.protected_files {
            decrypt_protected_file(
                &package,
                model,
                file,
                key.as_bytes(),
                &artifact_id,
                &nonce_prefix,
                manifest.encryption.record_plaintext_limit,
                &CancellationToken::default(),
            )?;
            created_outputs.push(file.output_path.as_str());
        }
        Ok(())
    })();

    if result.is_err() && remove_outputs(model, &created_outputs).is_err() {
        return Err(ProtectionError::CleanupFailed);
    }
    result
}

fn stage_public_metadata(
    package_root: &Path,
    model: &SafeDir,
    verified: &VerifiedManifest,
) -> Result<()> {
    let manifest = verified.manifest();
    manifest.validate()?;
    let package = SafeDir::open(package_root, false)?;
    model.require_empty()?;
    let mut created_outputs = Vec::new();
    let result = (|| {
        let mut safetensors_index = None;
        for file in &manifest.public_files {
            if let Some(bytes) = copy_public_file(&package, model, file)? {
                safetensors_index = Some(bytes);
            }
            created_outputs.push(file.output_path.as_str());
        }
        validate_safetensors_index(manifest, safetensors_index.as_deref())
    })();
    if result.is_err() && remove_outputs(model, &created_outputs).is_err() {
        return Err(ProtectionError::CleanupFailed);
    }
    result
}

fn materialize_protected_files(
    package_root: &Path,
    model: &SafeDir,
    verified: &VerifiedManifest,
    key: SecretDek,
    cancellation: &CancellationToken,
) -> Result<()> {
    let manifest = verified.manifest();
    let package = SafeDir::open(package_root, false)?;
    let artifact_id = manifest.artifact_id_bytes()?;
    let nonce_prefix = manifest.nonce_prefix_bytes()?;
    let mut created_outputs = Vec::new();
    let result = (|| {
        for file in &manifest.protected_files {
            decrypt_protected_file(
                &package,
                model,
                file,
                key.as_bytes(),
                &artifact_id,
                &nonce_prefix,
                manifest.encryption.record_plaintext_limit,
                cancellation,
            )?;
            created_outputs.push(file.output_path.as_str());
        }
        Ok(())
    })();
    if result.is_err() && remove_outputs(model, &created_outputs).is_err() {
        return Err(ProtectionError::CleanupFailed);
    }
    result
}

#[cfg(test)]
fn materialize_model_at(
    package_root: &Path,
    model_root: &Path,
    verified: &VerifiedManifest,
    key: SecretDek,
) -> Result<()> {
    let model = SafeDir::open(model_root, true)?;
    materialize_model(package_root, &model, verified, key)
}

/// Owns one verified plaintext model view and removes it on drop.
pub struct SecureModelSession {
    root: SafeDir,
    model: SafeDir,
    _owner_lock: File,
    name: String,
    path: PathBuf,
    manifest_digest: [u8; 32],
    public_staged: bool,
    active: bool,
}

impl SecureModelSession {
    /// Acquires the dedicated tmpfs root, removes stale owned sessions, and creates a fresh view.
    pub fn prepare(
        namespace: &str,
        authorized: &AuthorizedModel,
        process_memory_margin: u64,
    ) -> Result<Self> {
        let root_path = secure_model_root(namespace)?;
        Self::prepare_at(root_path, authorized, process_memory_margin, true)
    }

    fn prepare_at(
        root_path: PathBuf,
        authorized: &AuthorizedModel,
        process_memory_margin: u64,
        require_mount_root: bool,
    ) -> Result<Self> {
        let root = SafeDir::open(&root_path, true)?;
        if require_mount_root {
            root.require_mount_root()?;
        }
        let owner_lock = root.acquire_owner_lock()?;
        root.remove_stale_sessions()?;
        let required = required_tmpfs_bytes(authorized)?;
        root.require_capacity(required)?;
        require_memory_headroom(
            required
                .checked_add(process_memory_margin)
                .ok_or(ProtectionError::TmpfsInsufficient)?,
        )?;

        let name = uuid::Uuid::new_v4().simple().to_string();
        mkdirat(&root.0, name.as_str(), Mode::RWXU).map_err(io_error)?;
        let model = match root.open_child_dir(&name) {
            Ok(model) => model,
            Err(error) => {
                let _ = unlinkat(&root.0, name.as_str(), AtFlags::REMOVEDIR);
                return Err(error);
            }
        };
        let path = root_path.join(&name);
        Ok(Self {
            root,
            model,
            _owner_lock: owner_lock,
            name,
            path,
            manifest_digest: *authorized.manifest().digest(),
            public_staged: false,
            active: true,
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.path
    }

    /// Copies only signed, allowlisted public metadata into the private model view.
    pub fn stage_public_metadata(
        &mut self,
        package_root: &Path,
        authorized: &AuthorizedModel,
    ) -> Result<()> {
        if self.public_staged || self.manifest_digest != *authorized.manifest().digest() {
            return Err(ProtectionError::SessionConflict);
        }
        if let Err(error) = stage_public_metadata(package_root, &self.model, authorized.manifest())
        {
            self.remove()?;
            return Err(error);
        }
        self.public_staged = true;
        Ok(())
    }

    /// Consumes the owned DEK after public metadata and effective loader config are verified.
    pub fn materialize_protected(
        &mut self,
        package_root: &Path,
        authorized: &AuthorizedModel,
        key: SecretDek,
    ) -> Result<()> {
        if !self.public_staged || self.manifest_digest != *authorized.manifest().digest() {
            return Err(ProtectionError::SessionConflict);
        }
        if let Err(error) = materialize_protected_files(
            package_root,
            &self.model,
            authorized.manifest(),
            key,
            &CancellationToken::default(),
        ) {
            self.remove()?;
            return Err(error);
        }
        Ok(())
    }

    pub fn materialize_protected_cancellable(
        &mut self,
        package_root: &Path,
        authorized: &AuthorizedModel,
        key: SecretDek,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        if !self.public_staged || self.manifest_digest != *authorized.manifest().digest() {
            return Err(ProtectionError::SessionConflict);
        }
        if let Err(error) = materialize_protected_files(
            package_root,
            &self.model,
            authorized.manifest(),
            key,
            cancellation,
        ) {
            self.remove()?;
            return Err(error);
        }
        Ok(())
    }

    /// Convenience path for callers that have already completed both configuration gates.
    pub fn materialize(
        &mut self,
        package_root: &Path,
        authorized: &AuthorizedModel,
        key: SecretDek,
    ) -> Result<()> {
        self.stage_public_metadata(package_root, authorized)?;
        self.materialize_protected(package_root, authorized, key)
    }

    pub fn cleanup(mut self) -> Result<()> {
        self.remove()
    }

    fn remove(&mut self) -> Result<()> {
        if self.active {
            let mut entries = 0;
            remove_tree(&self.root.0, OsStr::new(&self.name), 0, &mut entries)?;
            self.active = false;
        }
        Ok(())
    }
}

impl Drop for SecureModelSession {
    fn drop(&mut self) {
        let _ = self.remove();
    }
}

struct SafeDir(OwnedFd);

impl SafeDir {
    fn open(path: &Path, require_private_tmpfs: bool) -> Result<Self> {
        if !path.is_absolute() {
            return Err(ProtectionError::InvalidPackage("root path"));
        }
        let opened = openat2(
            CWD,
            path,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
            ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
        );
        let fd = if require_private_tmpfs {
            opened.map_err(|_| ProtectionError::TmpfsInvalid)?
        } else {
            opened.map_err(io_error)?
        };
        if require_private_tmpfs {
            verify_private_tmpfs(&fd)?;
        }
        Ok(Self(fd))
    }

    fn open_child_dir(&self, name: &str) -> Result<Self> {
        let fd = openat2(
            &self.0,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
            SAFE_RESOLVE,
        )
        .map_err(|_| ProtectionError::TmpfsInvalid)?;
        verify_private_tmpfs(&fd)?;
        Ok(Self(fd))
    }

    fn open_regular(&self, path: &str, expected_size: u64) -> Result<File> {
        validate_relative_path(path)?;
        let fd = openat2(
            &self.0,
            path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
            SAFE_RESOLVE,
        )
        .map_err(|error| {
            if error == rustix::io::Errno::NOENT {
                ProtectionError::InvalidPackage("source file")
            } else {
                io_error(error)
            }
        })?;
        let stat = fstat(&fd).map_err(io_error)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
            || stat.st_nlink != 1
            || stat.st_size < 0
            || stat.st_size as u64 != expected_size
        {
            return Err(ProtectionError::InvalidPackage("source file"));
        }
        Ok(File::from(fd))
    }

    fn read_bounded(&self, path: &str, max_bytes: usize) -> Result<Vec<u8>> {
        validate_relative_path(path)?;
        let fd = openat2(
            &self.0,
            path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
            SAFE_RESOLVE,
        )
        .map_err(|error| {
            if error == rustix::io::Errno::NOENT {
                ProtectionError::InvalidPackage("metadata file")
            } else {
                io_error(error)
            }
        })?;
        let stat = fstat(&fd).map_err(io_error)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
            || stat.st_nlink != 1
            || stat.st_size <= 0
            || stat.st_size as u64 > max_bytes as u64
        {
            return Err(ProtectionError::InvalidPackage("metadata file"));
        }
        let file = File::from(fd);
        let mut bytes = Vec::with_capacity(stat.st_size as usize);
        file.take(max_bytes as u64 + 1).read_to_end(&mut bytes)?;
        if bytes.len() != stat.st_size as usize || bytes.len() > max_bytes {
            return Err(ProtectionError::InvalidPackage("metadata file"));
        }
        Ok(bytes)
    }

    fn require_empty(&self) -> Result<()> {
        let mut buffer = [MaybeUninit::uninit(); 4096];
        let mut entries = RawDir::new(&self.0, &mut buffer);
        while let Some(entry) = entries.next() {
            let entry = entry.map_err(io_error)?;
            let name = entry.file_name().to_bytes();
            if name != b"." && name != b".." {
                return Err(ProtectionError::InvalidPackage("model root not empty"));
            }
        }
        Ok(())
    }

    fn acquire_owner_lock(&self) -> Result<File> {
        let fd = openat2(
            &self.0,
            ".owner.lock",
            OFlags::RDWR | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::RUSR | Mode::WUSR,
            SAFE_RESOLVE,
        )
        .map_err(io_error)?;
        let stat = fstat(&fd).map_err(io_error)?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
            || stat.st_nlink != 1
            || stat.st_uid != geteuid().as_raw()
            || Mode::from_raw_mode(stat.st_mode).intersects(Mode::RWXG | Mode::RWXO)
        {
            return Err(ProtectionError::TmpfsInvalid);
        }
        flock(&fd, FlockOperation::NonBlockingLockExclusive)
            .map_err(|_| ProtectionError::SessionConflict)?;
        Ok(File::from(fd))
    }

    fn remove_stale_sessions(&self) -> Result<()> {
        let entries = read_names(&self.0)?;
        let mut removed = 0;
        for name in entries {
            if matches!(name.as_bytes(), b"." | b".." | b".owner.lock") {
                continue;
            }
            let text = name
                .to_str()
                .filter(|text| text.len() == 32)
                .filter(|text| {
                    text.bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
                .ok_or(ProtectionError::TmpfsInvalid)?;
            remove_tree(&self.0, OsStr::new(text), 0, &mut removed)?;
        }
        Ok(())
    }

    fn require_capacity(&self, required: u64) -> Result<()> {
        let stat = fstatvfs(&self.0).map_err(|_| ProtectionError::TmpfsInvalid)?;
        let available = stat
            .f_bavail
            .checked_mul(stat.f_frsize)
            .ok_or(ProtectionError::TmpfsInvalid)?;
        if available < required {
            return Err(ProtectionError::TmpfsInsufficient);
        }
        Ok(())
    }

    fn require_mount_root(&self) -> Result<()> {
        let stat = statx(
            &self.0,
            "",
            AtFlags::EMPTY_PATH | AtFlags::NO_AUTOMOUNT,
            StatxFlags::empty(),
        )
        .map_err(|_| ProtectionError::TmpfsInvalid)?;
        if !stat
            .stx_attributes_mask
            .contains(StatxAttributes::MOUNT_ROOT)
            || !stat.stx_attributes.contains(StatxAttributes::MOUNT_ROOT)
        {
            return Err(ProtectionError::TmpfsInvalid);
        }
        Ok(())
    }

    fn output_parent<'a>(&self, path: &'a str) -> Result<(OwnedFd, &'a OsStr)> {
        validate_relative_path(path)?;
        let path = Path::new(path);
        let file_name = path
            .file_name()
            .ok_or(ProtectionError::InvalidPackage("output path"))?;
        let mut parent = fcntl_dupfd_cloexec(&self.0, 0).map_err(io_error)?;
        if let Some(parent_path) = path.parent() {
            for component in parent_path.components() {
                let Component::Normal(name) = component else {
                    return Err(ProtectionError::InvalidPackage("output path"));
                };
                match mkdirat(&parent, name, Mode::RWXU) {
                    Ok(()) => {}
                    Err(rustix::io::Errno::EXIST) => {}
                    Err(error) => return Err(io_error(error)),
                }
                parent = openat2(
                    &parent,
                    name,
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
                    Mode::empty(),
                    SAFE_RESOLVE,
                )
                .map_err(io_error)?;
            }
        }
        Ok((parent, file_name))
    }
}

fn required_tmpfs_bytes(authorized: &AuthorizedModel) -> Result<u64> {
    let manifest = authorized.manifest().manifest();
    manifest
        .protected_files
        .iter()
        .map(|file| file.plaintext_size)
        .chain(manifest.public_files.iter().map(|file| file.size))
        .try_fold(0_u64, |total, size| {
            total
                .checked_add(size)
                .ok_or(ProtectionError::TmpfsInsufficient)
        })
}

fn require_memory_headroom(required: u64) -> Result<()> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let available_kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or(ProtectionError::TmpfsInvalid)?;
    let available = available_kib
        .checked_mul(1024)
        .ok_or(ProtectionError::TmpfsInvalid)?;
    if available < required {
        return Err(ProtectionError::TmpfsInsufficient);
    }

    let cgroup = std::fs::read_to_string("/proc/self/cgroup")?;
    let relative = cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::/"))
        .ok_or(ProtectionError::TmpfsInvalid)?;
    if !relative.is_empty()
        && relative
            .split('/')
            .any(|part| matches!(part, "" | "." | ".."))
    {
        return Err(ProtectionError::TmpfsInvalid);
    }
    let cgroup_root = Path::new("/sys/fs/cgroup");
    require_cgroup_headroom(cgroup_root, &cgroup_root.join(relative), required)?;
    Ok(())
}

fn require_cgroup_headroom(base: &Path, leaf: &Path, required: u64) -> Result<()> {
    let mut path = leaf;
    loop {
        let maximum = match std::fs::read_to_string(path.join("memory.max")) {
            Ok(maximum) => maximum,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && path != leaf => {
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        };
        if maximum.trim() != "max" {
            let maximum = maximum
                .trim()
                .parse::<u64>()
                .map_err(|_| ProtectionError::TmpfsInvalid)?;
            let current = std::fs::read_to_string(path.join("memory.current"))?
                .trim()
                .parse::<u64>()
                .map_err(|_| ProtectionError::TmpfsInvalid)?;
            if maximum.saturating_sub(current) < required {
                return Err(ProtectionError::TmpfsInsufficient);
            }
        }
        if path == base {
            return Ok(());
        }
        path = path
            .parent()
            .filter(|parent| parent.starts_with(base))
            .ok_or(ProtectionError::TmpfsInvalid)?;
    }
}

fn remove_tree(
    parent: &OwnedFd,
    name: &OsStr,
    depth: usize,
    entries_seen: &mut usize,
) -> Result<()> {
    if depth > MAX_CLEANUP_DEPTH || *entries_seen > MAX_CLEANUP_ENTRIES {
        return Err(ProtectionError::CleanupFailed);
    }
    let directory = openat2(
        parent,
        name,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
        SAFE_RESOLVE,
    )
    .map_err(|_| ProtectionError::CleanupFailed)?;
    let names = read_names(&directory).map_err(|_| ProtectionError::CleanupFailed)?;
    for child in names {
        if child.as_bytes() == b"." || child.as_bytes() == b".." {
            continue;
        }
        *entries_seen += 1;
        if *entries_seen > MAX_CLEANUP_ENTRIES {
            return Err(ProtectionError::CleanupFailed);
        }
        let stat = statat(&directory, &child, AtFlags::SYMLINK_NOFOLLOW)
            .map_err(|_| ProtectionError::CleanupFailed)?;
        if FileType::from_raw_mode(stat.st_mode) == FileType::Directory {
            remove_tree(&directory, &child, depth + 1, entries_seen)?;
        } else {
            unlinkat(&directory, &child, AtFlags::empty())
                .map_err(|_| ProtectionError::CleanupFailed)?;
        }
    }
    drop(directory);
    unlinkat(parent, name, AtFlags::REMOVEDIR).map_err(|_| ProtectionError::CleanupFailed)
}

fn read_names(directory: &OwnedFd) -> Result<Vec<OsString>> {
    let mut buffer = [MaybeUninit::uninit(); 4096];
    let mut raw = RawDir::new(directory, &mut buffer);
    let mut names = Vec::new();
    while let Some(entry) = raw.next() {
        if names.len() == MAX_CLEANUP_ENTRIES {
            return Err(ProtectionError::CleanupFailed);
        }
        let entry = entry.map_err(io_error)?;
        names.push(OsStr::from_bytes(entry.file_name().to_bytes()).to_owned());
    }
    Ok(names)
}

fn verify_private_tmpfs(fd: &OwnedFd) -> Result<()> {
    let fs = fstatfs(fd).map_err(|_| ProtectionError::TmpfsInvalid)?;
    let stat = fstat(fd).map_err(|_| ProtectionError::TmpfsInvalid)?;
    if fs.f_type as i64 != TMPFS_MAGIC
        || FileType::from_raw_mode(stat.st_mode) != FileType::Directory
        || stat.st_uid != geteuid().as_raw()
        || Mode::from_raw_mode(stat.st_mode).intersects(Mode::RWXG | Mode::RWXO)
    {
        return Err(ProtectionError::TmpfsInvalid);
    }
    Ok(())
}

fn copy_public_file(
    package: &SafeDir,
    model: &SafeDir,
    spec: &PublicFile,
) -> Result<Option<Vec<u8>>> {
    let mut source = package.open_regular(&spec.source_path, spec.size)?;
    let expected_hash = decode_lower_hex::<32>(&spec.sha256, "public sha256")?;
    let (parent, name) = model.output_parent(&spec.output_path)?;
    let partial = partial_name(name)?;
    let mut output = PartialOutput::create(&parent, &partial)?;
    let captured = if spec.output_path == "model.safetensors.index.json" {
        let bytes = read_exact_hashed(&mut source, spec.size, expected_hash)?;
        output.file_mut()?.write_all(&bytes)?;
        Some(bytes)
    } else {
        copy_exact_hashed(&mut source, output.file_mut()?, spec.size, expected_hash)?;
        None
    };
    output.finish(Ok(()), name)?;
    Ok(captured)
}

#[allow(clippy::too_many_arguments)]
fn decrypt_protected_file(
    package: &SafeDir,
    model: &SafeDir,
    spec: &ProtectedFile,
    key: &[u8; 32],
    artifact_id: &[u8; 16],
    nonce_prefix: &[u8; 4],
    record_plaintext_limit: u32,
    cancellation: &CancellationToken,
) -> Result<()> {
    let source = package.open_regular(&spec.container_path, spec.container_size)?;
    let (parent, name) = model.output_parent(&spec.output_path)?;
    let partial = partial_name(name)?;
    let mut output = PartialOutput::create(&parent, &partial)?;
    let result = decrypt_records(
        source,
        output.file_mut()?,
        key,
        artifact_id,
        nonce_prefix,
        spec,
        record_plaintext_limit,
        cancellation,
    )
    .map(|_| ());
    output.finish(result, name)
}

fn create_exclusive(parent: &OwnedFd, name: &OsStr) -> Result<File> {
    let fd = openat2(
        parent,
        name,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::RUSR | Mode::WUSR,
        SAFE_RESOLVE,
    )
    .map_err(io_error)?;
    Ok(File::from(fd))
}

struct PartialOutput<'a> {
    file: Option<File>,
    parent: &'a OwnedFd,
    name: &'a OsStr,
    active: bool,
}

impl<'a> PartialOutput<'a> {
    fn create(parent: &'a OwnedFd, name: &'a OsStr) -> Result<Self> {
        Ok(Self {
            file: Some(create_exclusive(parent, name)?),
            parent,
            name,
            active: true,
        })
    }

    fn file_mut(&mut self) -> Result<&mut File> {
        self.file.as_mut().ok_or(ProtectionError::CleanupFailed)
    }

    fn finish(mut self, result: Result<()>, final_name: &OsStr) -> Result<()> {
        let result = result.and_then(|()| self.file_mut()?.sync_all().map_err(Into::into));
        drop(self.file.take());
        if let Err(error) = result {
            self.remove()?;
            return Err(error);
        }
        if let Err(error) = renameat_with(
            self.parent,
            self.name,
            self.parent,
            final_name,
            RenameFlags::NOREPLACE,
        ) {
            self.remove()?;
            return Err(io_error(error));
        }
        self.active = false;
        Ok(())
    }

    fn remove(&mut self) -> Result<()> {
        unlinkat(self.parent, self.name, AtFlags::empty())
            .map_err(|_| ProtectionError::CleanupFailed)?;
        self.active = false;
        Ok(())
    }
}

impl Drop for PartialOutput<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = unlinkat(self.parent, self.name, AtFlags::empty());
        }
    }
}

fn copy_exact_hashed(
    source: &mut File,
    output: &mut File,
    expected_size: u64,
    expected_hash: [u8; 32],
) -> Result<()> {
    let mut remaining = expected_size;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let wanted = remaining.min(COPY_BUFFER_BYTES as u64) as usize;
        source
            .read_exact(&mut buffer[..wanted])
            .map_err(|_| ProtectionError::PackageIntegrityMismatch)?;
        output.write_all(&buffer[..wanted])?;
        hasher.update(&buffer[..wanted]);
        remaining -= wanted as u64;
    }
    if source.read(&mut buffer[..1])? != 0 || <[u8; 32]>::from(hasher.finalize()) != expected_hash {
        return Err(ProtectionError::PackageIntegrityMismatch);
    }
    Ok(())
}

fn read_exact_hashed(source: &mut File, size: u64, expected_hash: [u8; 32]) -> Result<Vec<u8>> {
    let size = usize::try_from(size).map_err(|_| ProtectionError::InvalidPackage("public size"))?;
    let mut bytes = vec![0; size];
    source
        .read_exact(&mut bytes)
        .map_err(|_| ProtectionError::PackageIntegrityMismatch)?;
    let mut trailing = [0_u8; 1];
    if source.read(&mut trailing)? != 0 || <[u8; 32]>::from(Sha256::digest(&bytes)) != expected_hash
    {
        return Err(ProtectionError::PackageIntegrityMismatch);
    }
    Ok(bytes)
}

fn partial_name(name: &OsStr) -> Result<std::ffi::OsString> {
    let name = name
        .to_str()
        .ok_or(ProtectionError::InvalidPackage("output path"))?;
    Ok(format!("{name}.partial").into())
}

fn remove_outputs(model: &SafeDir, paths: &[&str]) -> Result<()> {
    for path in paths {
        let (parent, name) = model
            .output_parent(path)
            .map_err(|_| ProtectionError::CleanupFailed)?;
        unlinkat(&parent, name, AtFlags::empty()).map_err(|_| ProtectionError::CleanupFailed)?;
    }
    Ok(())
}

fn io_error(_: rustix::io::Errno) -> ProtectionError {
    ProtectionError::Io
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as BASE64;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use rustix::io::{FdFlags, fcntl_getfd};
    use std::os::unix::fs::PermissionsExt;

    fn lower_hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        output
    }

    fn authorized_model() -> AuthorizedModel {
        let manifest = crate::Manifest {
            format: crate::format::FORMAT_NAME.to_string(),
            format_version: crate::format::FORMAT_VERSION,
            artifact_id: "00".repeat(16),
            customer_scope_id: "test-scope".to_string(),
            model: crate::ModelIdentity {
                model_id: "tiny".to_string(),
                model_version: "1".to_string(),
                framework: "safetensors".to_string(),
            },
            encryption: crate::Encryption {
                algorithm: "AES-256-GCM".to_string(),
                nonce_prefix: "00".repeat(4),
                tag_bits: 128,
                record_plaintext_limit: 32,
            },
            protected_files: vec![ProtectedFile {
                file_id: 1,
                container_path: "weights/model.safetensors.protected".to_string(),
                output_path: "model.safetensors".to_string(),
                container_size: 84,
                container_sha256: "11".repeat(32),
                plaintext_size: 32,
                plaintext_sha256: "22".repeat(32),
                record_count: 1,
                first_global_record_counter: 0,
            }],
            public_files: Vec::new(),
            runtime: crate::RuntimeRequirements {
                minimum_runtime_version: env!("CARGO_PKG_VERSION").to_string(),
                required_load_format: "safetensors".to_string(),
            },
        };
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let signer = Ed25519KeyPair::from_seed_unchecked(&[9; 32]).unwrap();
        let signature = serde_json::to_vec(&crate::SignatureEnvelope {
            algorithm: "Ed25519".to_string(),
            key_id: "test".to_string(),
            signature: BASE64.encode(
                signer
                    .sign(&crate::manifest_signature_payload(&bytes))
                    .as_ref(),
            ),
        })
        .unwrap();
        let verified = crate::verify_manifest(
            &bytes,
            &signature,
            "test",
            signer.public_key().as_ref().try_into().unwrap(),
        )
        .unwrap();
        let license = crate::License {
            format: crate::license::LICENSE_FORMAT.to_string(),
            format_version: crate::license::LICENSE_FORMAT_VERSION,
            license_id: "test-license".to_string(),
            customer_scope_id: "test-scope".to_string(),
            artifact_id: "00".repeat(16),
            manifest_sha256: lower_hex(verified.digest()),
            model_id: "tiny".to_string(),
            model_version: "1".to_string(),
            entitlement: crate::Entitlement {
                mode: "offline-perpetual".to_string(),
                generation: 1,
            },
            recipient: crate::TpmRecipient {
                kind: "tpm2".to_string(),
                profile: crate::license::TPM_PROFILE.to_string(),
                device_key_name: "000b".to_string() + &"11".repeat(32),
                device_public_key_sha256: "11".repeat(32),
                command_parameters_hash: "55".repeat(32),
                approved_policy_digest: "44".repeat(32),
                policy_ref: crate::license::TPM_POLICY_REF_HEX.to_string(),
                policy_authority_key_id: "test-policy".to_string(),
                policy_signature_algorithm: crate::license::TPM_POLICY_SIGNATURE.to_string(),
                policy_signature: BASE64.encode([6; 64]),
            },
            wrapped_dek: BASE64.encode([5; 256]),
        };
        AuthorizedModel::new(license, verified)
    }

    #[test]
    fn namespace_contract_is_unambiguous() {
        for valid in ["dynamo", "team-a.worker_1", "A1"] {
            assert_eq!(
                secure_model_root(valid).unwrap(),
                PathBuf::from(format!("/run/{valid}-models"))
            );
        }
        for invalid in ["", ".hidden", "trailing-", "a/b", "..", "mô-hình"] {
            assert!(validate_namespace(invalid).is_err(), "accepted {invalid:?}");
        }
        assert!(matches!(
            load_authorized_model(
                Path::new("/missing-package"),
                Path::new("/missing-license"),
                "same-key",
                &[1; 32],
                "same-key",
                &[1; 32],
            ),
            Err(ProtectionError::LicenseInvalid("trust domain"))
        ));
    }

    #[test]
    fn cgroup_headroom_honors_ancestor_limits() {
        let base = tempfile::tempdir().unwrap();
        let leaf = base.path().join("parent/leaf");
        std::fs::create_dir_all(&leaf).unwrap();
        for (path, maximum, current) in [
            (base.path(), "max", "900"),
            (leaf.parent().unwrap(), "1000", "950"),
            (leaf.as_path(), "max", "10"),
        ] {
            std::fs::write(path.join("memory.max"), maximum).unwrap();
            std::fs::write(path.join("memory.current"), current).unwrap();
        }
        assert!(matches!(
            require_cgroup_headroom(base.path(), &leaf, 100),
            Err(ProtectionError::TmpfsInsufficient)
        ));
        std::fs::write(leaf.parent().unwrap().join("memory.current"), "800").unwrap();
        assert!(require_cgroup_headroom(base.path(), &leaf, 100).is_ok());
    }

    #[test]
    fn tmpfs_verification_rejects_disk_and_symlink_roots() {
        let disk = tempfile::tempdir().unwrap();
        std::fs::set_permissions(disk.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(matches!(
            SafeDir::open(disk.path(), true),
            Err(ProtectionError::TmpfsInvalid)
        ));

        let link_parent = tempfile::tempdir().unwrap();
        let link = link_parent.path().join("linked");
        std::os::unix::fs::symlink(disk.path(), &link).unwrap();
        assert!(SafeDir::open(&link, false).is_err());
    }

    #[test]
    fn session_exclusively_owns_and_cleans_tmpfs_view() {
        if !Path::new("/dev/shm").is_dir() {
            return;
        }
        let root = tempfile::Builder::new()
            .prefix("dynamo-model-protection-root-")
            .tempdir_in("/dev/shm")
            .unwrap();
        std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let stale = root.path().join("0".repeat(32));
        std::fs::create_dir(&stale).unwrap();
        std::fs::write(stale.join("plaintext"), b"stale").unwrap();

        let authorized = authorized_model();
        let session =
            SecureModelSession::prepare_at(root.path().to_path_buf(), &authorized, 0, false)
                .unwrap();
        assert!(!stale.exists());
        assert!(session.model_path().is_dir());
        assert!(matches!(
            SecureModelSession::prepare_at(root.path().to_path_buf(), &authorized, 0, false),
            Err(ProtectionError::SessionConflict)
        ));
        let path = session.model_path().to_path_buf();
        drop(session);
        assert!(!path.exists());
    }

    #[test]
    fn materializes_public_and_authenticated_files_into_tmpfs() {
        let package = tempfile::tempdir().unwrap();
        let Some(tmpfs_parent) = Path::new("/dev/shm")
            .is_dir()
            .then(|| Path::new("/dev/shm"))
        else {
            return;
        };
        let model = tempfile::Builder::new()
            .prefix("dynamo-model-protection-test-")
            .tempdir_in(tmpfs_parent)
            .unwrap();
        std::fs::set_permissions(model.path(), std::fs::Permissions::from_mode(0o700)).unwrap();

        std::fs::create_dir(package.path().join("public")).unwrap();
        std::fs::create_dir(package.path().join("weights")).unwrap();
        let public = b"{}";
        std::fs::write(package.path().join("public/config.json"), public).unwrap();
        let plaintext = b"protected-model-bytes";
        let key = [0x42; 32];
        let artifact_id = [0x24; 16];
        let nonce_prefix = [1, 2, 3, 4];
        let mut container = Vec::new();
        let encrypted = crate::encrypt_records(
            plaintext.as_slice(),
            &mut container,
            &key,
            &artifact_id,
            &nonce_prefix,
            1,
            0,
            8,
        )
        .unwrap();
        std::fs::write(
            package.path().join("weights/model.safetensors.protected"),
            &container,
        )
        .unwrap();

        let manifest = crate::Manifest {
            format: crate::format::FORMAT_NAME.to_string(),
            format_version: crate::format::FORMAT_VERSION,
            artifact_id: lower_hex(&artifact_id),
            customer_scope_id: "test-scope".to_string(),
            model: crate::ModelIdentity {
                model_id: "tiny".to_string(),
                model_version: "1".to_string(),
                framework: "safetensors".to_string(),
            },
            encryption: crate::Encryption {
                algorithm: "AES-256-GCM".to_string(),
                nonce_prefix: lower_hex(&nonce_prefix),
                tag_bits: 128,
                record_plaintext_limit: 8,
            },
            protected_files: vec![ProtectedFile {
                file_id: 1,
                container_path: "weights/model.safetensors.protected".to_string(),
                output_path: "model.safetensors".to_string(),
                container_size: encrypted.container_size,
                container_sha256: lower_hex(&encrypted.container_sha256),
                plaintext_size: encrypted.plaintext_size,
                plaintext_sha256: lower_hex(&encrypted.plaintext_sha256),
                record_count: encrypted.record_count,
                first_global_record_counter: 0,
            }],
            public_files: vec![PublicFile {
                source_path: "public/config.json".to_string(),
                output_path: "config.json".to_string(),
                size: public.len() as u64,
                sha256: lower_hex(&Sha256::digest(public)),
                publish_to_model_card: true,
            }],
            runtime: crate::RuntimeRequirements {
                minimum_runtime_version: "1.5.0".to_string(),
                required_load_format: "safetensors".to_string(),
            },
        };

        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        assert!(
            validate_safetensors_index(
                &manifest,
                Some(br#"{"metadata":{},"weight_map":{"weight":"model.safetensors"}}"#)
            )
            .is_ok()
        );
        assert!(
            validate_safetensors_index(
                &manifest,
                Some(br#"{"weight_map":{"weight":"external.safetensors"}}"#)
            )
            .is_err()
        );
        let signer = Ed25519KeyPair::from_seed_unchecked(&[7_u8; 32]).unwrap();
        let envelope = serde_json::to_vec(&crate::SignatureEnvelope {
            algorithm: "Ed25519".to_string(),
            key_id: "package-test".to_string(),
            signature: BASE64.encode(
                signer
                    .sign(&crate::manifest_signature_payload(&manifest_bytes))
                    .as_ref(),
            ),
        })
        .unwrap();
        std::fs::write(
            package.path().join("model.protection.json"),
            &manifest_bytes,
        )
        .unwrap();
        std::fs::write(package.path().join("model.protection.sig"), &envelope).unwrap();
        let verified = load_verified_manifest(
            package.path(),
            "package-test",
            signer.public_key().as_ref().try_into().unwrap(),
        )
        .unwrap();

        let model_dir = SafeDir::open(model.path(), true).unwrap();
        stage_public_metadata(package.path(), &model_dir, &verified).unwrap();
        assert_eq!(
            std::fs::read(model.path().join("config.json")).unwrap(),
            public
        );
        assert!(!model.path().join("model.safetensors").exists());
        materialize_protected_files(
            package.path(),
            &model_dir,
            &verified,
            SecretDek::new(key),
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(
            std::fs::read(model.path().join("config.json")).unwrap(),
            public
        );
        assert_eq!(
            std::fs::read(model.path().join("model.safetensors")).unwrap(),
            plaintext
        );
        assert!(!model.path().join("config.json.partial").exists());

        container[RECORD_HEADER_BYTES_FOR_TEST + 1] ^= 1;
        std::fs::write(
            package.path().join("weights/model.safetensors.protected"),
            &container,
        )
        .unwrap();
        let failed_model = tempfile::Builder::new()
            .prefix("dynamo-model-protection-test-")
            .tempdir_in(tmpfs_parent)
            .unwrap();
        std::fs::set_permissions(failed_model.path(), std::fs::Permissions::from_mode(0o700))
            .unwrap();
        assert!(
            materialize_model_at(
                package.path(),
                failed_model.path(),
                &verified,
                SecretDek::new(key),
            )
            .is_err()
        );
        assert!(!failed_model.path().join("config.json").exists());
        assert!(
            !failed_model
                .path()
                .join("nested/model.safetensors.partial")
                .exists()
        );

        let collision_model = tempfile::Builder::new()
            .prefix("dynamo-model-protection-test-")
            .tempdir_in(tmpfs_parent)
            .unwrap();
        std::fs::set_permissions(
            collision_model.path(),
            std::fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        std::fs::write(
            collision_model.path().join("undeclared.safetensors"),
            b"keep",
        )
        .unwrap();
        assert!(
            materialize_model_at(
                package.path(),
                collision_model.path(),
                &verified,
                SecretDek::new(key),
            )
            .is_err()
        );
        assert_eq!(
            std::fs::read(collision_model.path().join("undeclared.safetensors")).unwrap(),
            b"keep"
        );
    }

    #[test]
    fn rejects_special_inputs_and_cleans_failed_publication() {
        let package = tempfile::tempdir().unwrap();
        let fifo = package.path().join("model.protection.json");
        rustix::fs::mkfifoat(CWD, &fifo, Mode::RUSR | Mode::WUSR).unwrap();
        assert!(matches!(
            load_verified_manifest(package.path(), "test", &[0; 32]),
            Err(ProtectionError::InvalidPackage("metadata file"))
        ));

        let Some(tmpfs_parent) = Path::new("/dev/shm")
            .is_dir()
            .then(|| Path::new("/dev/shm"))
        else {
            return;
        };
        let model = tempfile::Builder::new()
            .prefix("dynamo-model-protection-test-")
            .tempdir_in(tmpfs_parent)
            .unwrap();
        std::fs::set_permissions(model.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let model = SafeDir::open(model.path(), true).unwrap();
        let (parent, _) = model.output_parent("nested/output").unwrap();
        assert!(fcntl_getfd(&parent).unwrap().contains(FdFlags::CLOEXEC));

        if let Ok(full) = std::fs::OpenOptions::new().write(true).open("/dev/full") {
            let partial_name = OsStr::new("output.partial");
            let mut partial = PartialOutput::create(&parent, partial_name).unwrap();
            partial.file = Some(full);
            assert!(partial.finish(Ok(()), OsStr::new("output")).is_err());
            assert!(
                openat2(
                    &parent,
                    partial_name,
                    OFlags::RDONLY | OFlags::CLOEXEC,
                    Mode::empty(),
                    SAFE_RESOLVE,
                )
                .is_err()
            );
        }
    }

    const RECORD_HEADER_BYTES_FOR_TEST: usize = 36;
}
