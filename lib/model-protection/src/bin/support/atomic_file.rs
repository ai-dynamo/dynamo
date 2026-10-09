// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{Error, ErrorKind, Result, Write};
use std::path::Path;

use rustix::fs::{
    AtFlags, CWD, Mode, OFlags, RenameFlags, ResolveFlags, fstat, fsync, openat, openat2,
    renameat_with, unlinkat,
};

/// Publish into an owner-only directory through one pinned parent FD.
/// A failure after rename retains the final file for operator recovery.
pub fn publish_new(path: &Path, bytes: &[u8]) -> Result<()> {
    if !path.is_absolute() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "absolute path required",
        ));
    }
    let name = path.file_name().ok_or(ErrorKind::InvalidInput)?;
    let parent = openat2(
        CWD,
        path.parent().ok_or(ErrorKind::InvalidInput)?,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
    )?;
    let stat = fstat(&parent)?;
    if stat.st_uid != rustix::process::geteuid().as_raw() || stat.st_mode & 0o077 != 0 {
        return Err(Error::new(
            ErrorKind::PermissionDenied,
            "private parent required",
        ));
    }
    let partial = format!(".enrollment-partial-{}", uuid::Uuid::new_v4().simple());
    let fd = openat(
        &parent,
        partial.as_str(),
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::RUSR | Mode::WUSR,
    )?;
    let mut file = File::from(fd);
    let result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        renameat_with(
            &parent,
            partial.as_str(),
            &parent,
            name,
            RenameFlags::NOREPLACE,
        )?;
        fsync(&parent)?;
        Ok(())
    })();
    if result.is_err() {
        // Only our create_new partial is eligible for cleanup, never the final target.
        let _ = unlinkat(&parent, partial.as_str(), AtFlags::empty());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};

    #[test]
    fn publishes_owner_only_without_overwrite_or_symlink_parents() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = directory.path().join("request.json");
        publish_new(&path, b"first").unwrap();
        assert!(publish_new(&path, b"second").is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let alias = directory.path().join("alias");
        symlink(directory.path(), &alias).unwrap();
        assert!(publish_new(&alias.join("other"), b"data").is_err());
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(publish_new(&directory.path().join("public-parent"), b"data").is_err());
    }
}
