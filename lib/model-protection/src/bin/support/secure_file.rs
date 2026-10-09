// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{Error, ErrorKind, Read, Result};
use std::path::Path;

use rustix::fs::{CWD, FileType, Mode, OFlags, ResolveFlags, fstat, openat2};
use rustix::process::geteuid;

pub fn open_regular(path: &Path, maximum: u64, private: bool) -> Result<File> {
    if !path.is_absolute() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "absolute path required",
        ));
    }
    let fd = openat2(
        CWD,
        path,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
        ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS,
    )?;
    let stat = fstat(&fd)?;
    let mode = Mode::from_raw_mode(stat.st_mode);
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_nlink != 1
        || stat.st_size < 0
        || stat.st_size as u64 > maximum
        || (private
            && (stat.st_uid != geteuid().as_raw() || mode.intersects(Mode::RWXG | Mode::RWXO)))
    {
        return Err(Error::new(ErrorKind::InvalidData, "invalid file"));
    }
    Ok(File::from(fd))
}

pub fn read_bounded(path: &Path, maximum: u64, private: bool) -> Result<Vec<u8>> {
    let mut file = open_regular(path, maximum, private)?;
    let expected = file.metadata()?.len();
    let mut bytes = Vec::with_capacity(expected as usize);
    file.by_ref().take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != expected || bytes.len() as u64 > maximum {
        return Err(Error::new(ErrorKind::InvalidData, "file changed"));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::symlink;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn rejects_symlinks_and_oversized_files() {
        let directory = tempdir().unwrap();
        let file = directory.path().join("key");
        std::fs::write(&file, b"12345").unwrap();

        let link = directory.path().join("key-link");
        symlink(&file, &link).unwrap();
        assert!(read_bounded(&link, 5, false).is_err());
        assert!(read_bounded(&file, 4, false).is_err());
        assert_eq!(read_bounded(&file, 5, false).unwrap(), b"12345");
    }
}
