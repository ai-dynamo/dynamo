// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Unix socket the consumer serves its local API on.
//!
//! The socket lives in a directory that only the pod's containers mount, an
//! `emptyDir`, so the conductor reaches it without TLS or credentials. The
//! rules are those of the conductor's own OpenEngine socket.

use std::io::ErrorKind;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use tokio::net::UnixListener;

/// `sockaddr_un.sun_path` holds this many bytes, including the terminating NUL.
const MAX_PATH_BYTES: usize = 108;
/// Owner and group may connect; nobody else may.
const MODE: u32 = 0o660;

/// Fails unless `path` can name the socket: absolute and short enough to bind.
pub(crate) fn check_socket_path(path: &Path) -> Result<(), String> {
    if !path.is_absolute() || path.as_os_str().len() >= MAX_PATH_BYTES {
        return Err(format!(
            "the socket path {} must be absolute and shorter than {MAX_PATH_BYTES} bytes",
            path.display()
        ));
    }
    Ok(())
}

/// A socket file this process bound. Dropping it after its listener removes
/// the file, unless something else has replaced it by then.
pub(crate) struct SocketFile {
    path: PathBuf,
    /// The device and inode of the socket this process bound. A file system
    /// may give a socket that replaces it the same inode, so a match alone does
    /// not prove the file is still this process's.
    bound: (u64, u64),
}

/// What was at the socket path.
enum Existing {
    Missing,
    /// A socket no process served, now removed.
    Removed,
    Served,
    NotSocket,
}

/// Binds a listener at `path`, creating its directory if needed and replacing
/// a socket that a killed process left behind. Refuses a path that a live
/// process serves or that is not a socket.
pub(crate) async fn bind(path: &Path) -> anyhow::Result<(UnixListener, SocketFile)> {
    check_socket_path(path).map_err(anyhow::Error::msg)?;
    let shown = path.display();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create the directory of {shown}"))?;
    }
    match remove_unserved(path).with_context(|| format!("failed to replace {shown}"))? {
        Existing::Missing => {}
        Existing::Removed => tracing::info!(path = %shown, "Removed a stale LPX topology socket"),
        Existing::Served => bail!("another process serves the LPX topology API on {shown}"),
        Existing::NotSocket => bail!("{shown} exists and is not a socket"),
    }
    let listener = UnixListener::bind(path)
        .with_context(|| format!("failed to bind the LPX topology socket {shown}"))?;
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("failed to inspect the LPX topology socket {shown}"))?;
    // From here, dropping the file removes it once the listener is closed.
    let file = SocketFile {
        path: path.to_path_buf(),
        bound: (metadata.dev(), metadata.ino()),
    };
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(MODE))
        .with_context(|| format!("failed to limit {shown} to its owner and group"))?;
    Ok((listener, file))
}

impl SocketFile {
    /// Whether the path still names the socket this process bound: the same
    /// file, still a socket, and served by no process now that this process's
    /// listener is closed.
    fn is_ours(&self) -> std::io::Result<bool> {
        let metadata = std::fs::symlink_metadata(&self.path)?;
        Ok(metadata.file_type().is_socket()
            && (metadata.dev(), metadata.ino()) == self.bound
            && !is_served(&self.path)?)
    }
}

impl Drop for SocketFile {
    fn drop(&mut self) {
        let path = self.path.display();
        match self.is_ours() {
            Ok(true) => match std::fs::remove_file(&self.path) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => {
                    tracing::warn!(%path, %error, "Failed to remove the LPX topology socket")
                }
            },
            Ok(false) => tracing::warn!(%path, "The LPX topology socket was replaced; leaving it"),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                tracing::warn!(%path, %error, "Failed to inspect the LPX topology socket")
            }
        }
    }
}

/// Removes the socket at `path` unless a process serves it or it is not a
/// socket.
fn remove_unserved(path: &Path) -> std::io::Result<Existing> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_socket() => return Ok(Existing::NotSocket),
        Ok(_) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Existing::Missing),
        Err(error) => return Err(error),
    }
    if is_served(path)? {
        return Ok(Existing::Served);
    }
    match std::fs::remove_file(path) {
        Ok(()) => Ok(Existing::Removed),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(Existing::Missing),
        Err(error) => Err(error),
    }
}

/// Whether a process accepts connections on the socket at `path`. The probe
/// never blocks: a listener whose backlog is full is serving too.
fn is_served(path: &Path) -> std::io::Result<bool> {
    let probe = socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None)?;
    probe.set_nonblocking(true)?;
    match probe.connect(&socket2::SockAddr::unix(path)?) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::WouldBlock => Ok(true),
        Err(error) if error.kind() == ErrorKind::ConnectionRefused => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::{FileTypeExt, PermissionsExt};
    use std::os::unix::net::{UnixListener as StdListener, UnixStream};
    use std::path::Path;

    use super::{MODE, bind, check_socket_path};

    fn is_socket(path: &Path) -> bool {
        std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_socket())
    }

    #[tokio::test]
    async fn binding_creates_the_directory_limits_the_socket_and_removes_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing/topology.sock");

        let (listener, file) = bind(&path).await.unwrap();

        assert!(is_socket(&path));
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, MODE, "{mode:o}");
        UnixStream::connect(&path).unwrap();
        drop(listener);
        drop(file);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn a_stale_socket_is_replaced_but_a_served_one_is_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("topology.sock");
        // A killed process left its socket behind.
        drop(StdListener::bind(&path).unwrap());
        assert!(is_socket(&path));

        let (listener, file) = bind(&path).await.unwrap();
        let error = bind(&path).await.err().expect("a served socket is refused");

        assert!(
            error.to_string().contains("another process serves"),
            "{error}"
        );
        UnixStream::connect(&path).expect("the served socket is left in place");
        drop((listener, file));
    }

    #[tokio::test]
    async fn a_file_that_is_not_a_socket_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("topology.sock");
        std::fs::write(&path, b"not a socket").unwrap();

        let error = bind(&path).await.err().expect("a regular file is refused");

        assert!(error.to_string().contains("is not a socket"), "{error}");
        assert_eq!(std::fs::read(&path).unwrap(), b"not a socket");
    }

    #[tokio::test]
    async fn dropping_the_file_leaves_a_path_something_else_took_over() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("topology.sock");

        // Another process moved the socket away and bound its own. The old
        // socket keeps its inode, so the new one cannot reuse it.
        let (listener, file) = bind(&path).await.unwrap();
        drop(listener);
        std::fs::rename(&path, dir.path().join("old.sock")).unwrap();
        let other = StdListener::bind(&path).unwrap();
        drop(file);
        assert!(is_socket(&path), "a socket another process bound is left");
        drop(other);
        std::fs::remove_file(&path).unwrap();

        // A served socket at the inode this process bound. On ext4 or
        // overlayfs, a socket bound after a removal gets the freed inode; the
        // listener left open stands in for that other process's.
        let (listener, file) = bind(&path).await.unwrap();
        let other = listener.into_std().unwrap();
        drop(file);
        UnixStream::connect(&path).expect("a socket another process serves is left");
        drop(other);
        std::fs::remove_file(&path).unwrap();

        let (listener, file) = bind(&path).await.unwrap();
        drop(listener);
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"not a socket").unwrap();
        drop(file);
        assert_eq!(std::fs::read(&path).unwrap(), b"not a socket");
    }

    #[tokio::test]
    async fn unusable_paths_are_refused() {
        let longest = format!("/{}", "d".repeat(106));
        check_socket_path(Path::new(&longest)).expect("107 bytes and a NUL fit");
        let long = format!("/{}", "d".repeat(107));
        for path in ["topology.sock", "run/topology.sock", long.as_str()] {
            let error = check_socket_path(Path::new(path)).expect_err(path);
            assert!(error.contains("must be absolute"), "{path}: {error}");
            let error = bind(Path::new(path)).await.err().expect(path);
            assert!(
                error.to_string().contains("must be absolute"),
                "{path}: {error}"
            );
        }
    }
}
