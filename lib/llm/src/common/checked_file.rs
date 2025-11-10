// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    fmt::Display,
    path::{Path, PathBuf},
    str::FromStr,
};

use either::Either;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, Visitor},
    ser::SerializeStruct as _,
};
use url::Url;

#[derive(Clone, Debug)]
pub struct CheckedFile {
    /// Either a path on local disk or a remote URL (usually nats object store)
    path: Either<PathBuf, Url>,

    /// Checksum of the contents of path
    checksum: Checksum,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Checksum {
    /// The checksum is a hex encoded string of the file's content
    hash: String,

    /// Checksum algorithm
    algorithm: CryptographicHashMethods,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Eq, PartialEq)]
pub enum CryptographicHashMethods {
    #[serde(rename = "blake3")]
    BLAKE3,
}

impl CheckedFile {
    pub fn from_disk<P: Into<PathBuf>>(filepath: P) -> anyhow::Result<Self> {
        let path: PathBuf = filepath.into();
        if !path.exists() {
            anyhow::bail!("File not found: {}", path.display());
        }
        if !path.is_file() {
            anyhow::bail!("Not a file: {}", path.display());
        }
        let hash = b3sum(&path)?;

        Ok(CheckedFile {
            path: Either::Left(path),
            checksum: Checksum::blake3(hash),
        })
    }

    /// Replace the local disk path with a remote URL.
    /// Just updates the field, doesn't move any files.
    pub fn move_to_url(&mut self, u: url::Url) {
        self.path = Either::Right(u);
    }

    /// Replace a remove URL with local disk path.
    /// Just updates the field, doesn't move any files.
    pub fn move_to_disk<P: Into<PathBuf>>(&mut self, p: P) {
        self.path = Either::Left(p.into());
    }

    pub fn path(&self) -> Option<&Path> {
        match self.path.as_ref() {
            Either::Left(p) => Some(p),
            Either::Right(_) => None,
        }
    }

    pub fn url(&self) -> Option<&Url> {
        match self.path.as_ref() {
            Either::Left(_) => None,
            Either::Right(u) => Some(u),
        }
    }

    pub fn is_nats_url(&self) -> bool {
        matches!(self.path.as_ref(), Either::Right(u) if u.scheme() == "nats")
    }

    pub fn checksum(&self) -> &Checksum {
        &self.checksum
    }

    /// Does the given file checksum to the same value as this CheckedFile?
    pub fn checksum_matches<P: AsRef<Path> + std::fmt::Debug>(&self, disk_file: P) -> bool {
        match b3sum(&disk_file) {
            Ok(h) => Checksum::blake3(h) == self.checksum,
            Err(error) => {
                tracing::error!(disk_file = %disk_file.as_ref().display(), checked_file = self.to_string(), %error, "Checksum does not match");
                false
            }
        }
    }

    /// Is the CheckedFile a path on disk that exists?
    pub fn is_local(&self) -> bool {
        match self.path.as_ref() {
            Either::Left(path) => path.exists(),
            Either::Right(_) => false, // is a Url
        }
    }

    /// Keep the filename but change it's containing directory to `dir`.
    /// This is used to point at a model file (e.g. `tokenizer.json`) in the HF cache dir.
    pub fn update_dir(&mut self, dir: &Path) {
        match self.path.as_mut() {
            Either::Left(path) => {
                if let Some(file_name) = path.file_name() {
                    let mut new_path = PathBuf::from(dir);
                    new_path.push(file_name);
                    *path = new_path;
                }
            }
            Either::Right(_) => tracing::warn!("Cannot update directory on URL"),
        }
    }
}

impl Display for CheckedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = match &self.path {
            Either::Left(local) => local.display().to_string(),
            Either::Right(url) => url.to_string(),
        };
        write!(f, "({p}, {})", self.checksum)
    }
}

impl Serialize for CheckedFile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut cf = serializer.serialize_struct("CheckedFile", 2)?;
        match &self.path {
            Either::Left(path) => cf.serialize_field("path", &path)?,
            Either::Right(url) => cf.serialize_field("path", &url)?,
        };
        cf.serialize_field("checksum", &self.checksum)?;
        cf.end()
    }
}

/// Internal type to simplify deserializing
#[derive(Deserialize)]
struct WireCheckedFile {
    path: String,
    checksum: Checksum,
}

// Convert from the temporary struct to CheckedFile with path type logic.
impl From<WireCheckedFile> for CheckedFile {
    fn from(temp: WireCheckedFile) -> Self {
        // Try to parse as a URL; if successful, use Either::Right(Url), else use Either::Left(PathBuf).
        match Url::parse(&temp.path) {
            Ok(url) => CheckedFile {
                path: Either::Right(url),
                checksum: temp.checksum,
            },
            Err(_) => CheckedFile {
                path: Either::Left(PathBuf::from(temp.path)),
                checksum: temp.checksum,
            },
        }
    }
}

// Implement Deserialize for CheckedFile using the temporary struct.
impl<'de> Deserialize<'de> for CheckedFile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize into WireCheckedFile, then convert to CheckedFile.
        let temp = WireCheckedFile::deserialize(deserializer)?;
        Ok(CheckedFile::from(temp))
    }
}

fn b3sum<T: AsRef<Path> + std::fmt::Debug>(path: T) -> anyhow::Result<String> {
    let path = path.as_ref();
    let metadata = std::fs::metadata(path)?;
    let filesize = metadata.len();
    let mut hasher = blake3::Hasher::new();

    if filesize > 128_000 {
        // multithreaded. blake3 recommend this above 128 KiB.
        hasher.update_mmap_rayon(path)?;
    } else {
        // Uses mmap above 16 KiB, normal load otherwise.
        hasher.update_mmap(path)?;
    }

    let hash = hasher.finalize();
    Ok(hash.to_string())
}

impl Checksum {
    pub fn blake3(hash: impl Into<String>) -> Self {
        Self::new(hash, CryptographicHashMethods::BLAKE3)
    }

    pub fn new(hash: impl Into<String>, algorithm: CryptographicHashMethods) -> Self {
        Self {
            hash: hash.into(),
            algorithm,
        }
    }
}

impl Serialize for Checksum {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serialized_str = format!("{}:{}", self.algorithm, self.hash);
        serializer.serialize_str(&serialized_str)
    }
}

impl<'de> Deserialize<'de> for Checksum {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ChecksumVisitor;

        impl Visitor<'_> for ChecksumVisitor {
            type Value = Checksum;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a string in the format `{algo}:{hash}`")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let parts: Vec<&str> = value.split(':').collect();
                if parts.len() != 2 {
                    return Err(de::Error::invalid_value(de::Unexpected::Str(value), &self));
                }

                let algorithm = parts[0].parse().map_err(|_| {
                    de::Error::invalid_value(de::Unexpected::Str(parts[0]), &"invalid algorithm")
                })?;

                Ok(Checksum::new(parts[1], algorithm))
            }
        }

        deserializer.deserialize_str(ChecksumVisitor)
    }
}

impl TryFrom<&str> for Checksum {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let parts: Vec<&str> = value.split(':').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid checksum format; expect `algo:hash`; got: {value}");
        }

        let algo = match parts[0] {
            "blake3" => CryptographicHashMethods::BLAKE3,
            _ => {
                anyhow::bail!("Unsupported cryptographic hash method: {}", parts[0]);
            }
        };

        Ok(Checksum::new(parts[1], algo))
    }
}

impl Default for Checksum {
    fn default() -> Self {
        Self {
            hash: "".to_string(),
            algorithm: CryptographicHashMethods::BLAKE3,
        }
    }
}

impl FromStr for CryptographicHashMethods {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "blake3" => Ok(CryptographicHashMethods::BLAKE3),
            _ => Err(format!("Unsupported algorithm: {}", s)),
        }
    }
}

impl Display for CryptographicHashMethods {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CryptographicHashMethods::BLAKE3 => write!(f, "blake3"),
        }
    }
}

impl Display for Checksum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.algorithm, self.hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_blake3() {
        let checksum = Checksum::blake3("a12c3d4");

        let serialized = serde_json::to_string(&checksum).unwrap();
        assert_eq!(serialized.trim(), "\"blake3:a12c3d4\"");
    }

    #[test]
    fn test_deserialization_blake3() {
        let s = "\"blake3:abcd1234\"";
        let deserialized: Checksum = serde_json::from_str(s).unwrap();

        assert_eq!(deserialized.algorithm, CryptographicHashMethods::BLAKE3);
        assert_eq!(deserialized.hash, "abcd1234");
    }

    #[test]
    fn test_deserialization_invalid_format() {
        let s = "\"invalidformat\"";
        let result: Result<Checksum, _> = serde_json::from_str(s);

        assert!(result.is_err());

        let s = "\"blake3:invalid:format\"";
        let result: Result<Checksum, _> = serde_json::from_str(s);

        assert!(result.is_err());
    }

    #[test]
    fn test_checked_file_from_disk() {
        let root = env!("CARGO_MANIFEST_DIR"); // ${WORKSPACE}/lib/llm
        let full_path = format!("{root}/tests/data/sample-models/TinyLlama_v1.1/config.json");
        let cf = CheckedFile::from_disk(full_path).unwrap();
        let expected =
            Checksum::blake3("62bc124be974d3a25db05bedc99422660c26715e5bbda0b37d14bd84a0c65ab2");
        assert_eq!(expected, *cf.checksum());
    }
}
