// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{Read, Write};

use ring::aead::{self, Aad, LessSafeKey, Nonce, UnboundKey};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::format::{MAX_RECORD_PLAINTEXT, ProtectedFile, decode_lower_hex};
use crate::{CancellationToken, ProtectionError, Result};

const RECORD_MAGIC: &[u8; 8] = b"MPROTV1\0";
const RECORD_VERSION: u16 = 1;
const RECORD_HEADER_BYTES: usize = 36;
const TAG_BYTES: usize = 16;
const AAD_DOMAIN: &[u8] = b"model-protection-record-v1\0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncryptResult {
    pub record_count: u32,
    pub container_size: u64,
    pub container_sha256: [u8; 32],
    pub plaintext_size: u64,
    pub plaintext_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecryptResult {
    pub record_count: u32,
    pub plaintext_size: u64,
    pub plaintext_sha256: [u8; 32],
}

#[allow(clippy::too_many_arguments)]
pub fn encrypt_records<R: Read, W: Write>(
    mut plaintext: R,
    mut output: W,
    key: &[u8; 32],
    artifact_id: &[u8; 16],
    nonce_prefix: &[u8; 4],
    file_id: u32,
    first_global_counter: u64,
    record_plaintext_limit: u32,
) -> Result<EncryptResult> {
    if file_id == 0 || record_plaintext_limit == 0 || record_plaintext_limit > MAX_RECORD_PLAINTEXT
    {
        return Err(ProtectionError::InvalidPackage("record parameters"));
    }
    let cipher = LessSafeKey::new(
        UnboundKey::new(&aead::AES_256_GCM, key)
            .map_err(|_| ProtectionError::InvalidPackage("dek size"))?,
    );
    let mut plaintext_hasher = Sha256::new();
    let mut container_hasher = Sha256::new();
    let mut plaintext_size = 0_u64;
    let mut container_size = 0_u64;
    let mut record_index = 0_u32;

    loop {
        let mut buffer = Zeroizing::new(vec![0_u8; record_plaintext_limit as usize]);
        let length = read_record(&mut plaintext, &mut buffer)?;
        if length == 0 {
            break;
        }
        buffer.truncate(length);
        plaintext_hasher.update(&buffer);
        plaintext_size = plaintext_size
            .checked_add(length as u64)
            .ok_or(ProtectionError::InvalidPackage("plaintext size"))?;
        let global_counter = first_global_counter
            .checked_add(u64::from(record_index))
            .ok_or(ProtectionError::InvalidPackage("record counter"))?;
        let length_u32 =
            u32::try_from(length).map_err(|_| ProtectionError::InvalidPackage("record length"))?;
        let header = encode_header(
            file_id,
            record_index,
            global_counter,
            length_u32,
            length_u32,
        );
        let aad = encode_aad(
            artifact_id,
            file_id,
            record_index,
            global_counter,
            length_u32,
        );
        let nonce_bytes = encode_nonce(nonce_prefix, global_counter);
        let tag = cipher
            .seal_in_place_separate_tag(
                Nonce::assume_unique_for_key(nonce_bytes),
                Aad::from(&aad),
                &mut buffer,
            )
            .map_err(|_| ProtectionError::InvalidPackage("encryption failed"))?;

        write_hashed(&mut output, &mut container_hasher, &header)?;
        write_hashed(&mut output, &mut container_hasher, &buffer)?;
        write_hashed(&mut output, &mut container_hasher, tag.as_ref())?;
        container_size = container_size
            .checked_add((RECORD_HEADER_BYTES + length + TAG_BYTES) as u64)
            .ok_or(ProtectionError::InvalidPackage("container size"))?;
        record_index = record_index
            .checked_add(1)
            .ok_or(ProtectionError::InvalidPackage("record count"))?;
    }

    if record_index == 0 {
        return Err(ProtectionError::InvalidPackage("empty protected file"));
    }
    output.flush()?;
    Ok(EncryptResult {
        record_count: record_index,
        container_size,
        container_sha256: container_hasher.finalize().into(),
        plaintext_size,
        plaintext_sha256: plaintext_hasher.finalize().into(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decrypt_records<R: Read, W: Write>(
    mut container: R,
    mut plaintext: W,
    key: &[u8; 32],
    artifact_id: &[u8; 16],
    nonce_prefix: &[u8; 4],
    manifest_file: &ProtectedFile,
    record_plaintext_limit: u32,
    cancellation: &CancellationToken,
) -> Result<DecryptResult> {
    if manifest_file.record_count == 0
        || record_plaintext_limit == 0
        || record_plaintext_limit > MAX_RECORD_PLAINTEXT
    {
        return Err(ProtectionError::InvalidPackage("record parameters"));
    }
    let expected_container_hash =
        decode_lower_hex::<32>(&manifest_file.container_sha256, "container sha256")?;
    let expected_plaintext_hash =
        decode_lower_hex::<32>(&manifest_file.plaintext_sha256, "plaintext sha256")?;
    let cipher = LessSafeKey::new(
        UnboundKey::new(&aead::AES_256_GCM, key)
            .map_err(|_| ProtectionError::InvalidPackage("dek size"))?,
    );
    let mut container_hasher = Sha256::new();
    let mut plaintext_hasher = Sha256::new();
    let mut container_size = 0_u64;
    let mut plaintext_size = 0_u64;

    for expected_index in 0..manifest_file.record_count {
        if cancellation.is_cancelled() {
            return Err(ProtectionError::MaterializationCancelled);
        }
        let mut header = [0_u8; RECORD_HEADER_BYTES];
        read_exact_hashed(&mut container, &mut container_hasher, &mut header)?;
        container_size = container_size
            .checked_add(RECORD_HEADER_BYTES as u64)
            .ok_or(ProtectionError::InvalidPackage("container size"))?;
        let parsed = decode_header(&header)?;
        let expected_counter = manifest_file
            .first_global_record_counter
            .checked_add(u64::from(expected_index))
            .ok_or(ProtectionError::InvalidPackage("record counter"))?;
        let expected_offset = u64::from(expected_index)
            .checked_mul(u64::from(record_plaintext_limit))
            .ok_or(ProtectionError::InvalidPackage("plaintext size"))?;
        let expected_length = manifest_file
            .plaintext_size
            .checked_sub(expected_offset)
            .ok_or(ProtectionError::InvalidPackage("record length"))?
            .min(u64::from(record_plaintext_limit)) as u32;
        if parsed.file_id != manifest_file.file_id
            || parsed.record_index != expected_index
            || parsed.global_counter != expected_counter
            || parsed.plaintext_length == 0
            || parsed.plaintext_length != expected_length
            || parsed.ciphertext_length != parsed.plaintext_length
        {
            return Err(ProtectionError::InvalidPackage("record header"));
        }

        let mut buffer = Zeroizing::new(vec![0_u8; parsed.ciphertext_length as usize]);
        read_exact_hashed(&mut container, &mut container_hasher, &mut buffer)?;
        let mut tag_bytes = Zeroizing::new([0_u8; TAG_BYTES]);
        read_exact_hashed(&mut container, &mut container_hasher, &mut tag_bytes[..])?;
        container_size = container_size
            .checked_add(buffer.len() as u64 + TAG_BYTES as u64)
            .ok_or(ProtectionError::InvalidPackage("container size"))?;

        let aad = encode_aad(
            artifact_id,
            parsed.file_id,
            parsed.record_index,
            parsed.global_counter,
            parsed.plaintext_length,
        );
        let nonce_bytes = encode_nonce(nonce_prefix, parsed.global_counter);
        if cipher
            .open_in_place_separate_tag(
                Nonce::assume_unique_for_key(nonce_bytes),
                Aad::from(&aad),
                aead::Tag::from(*tag_bytes),
                &mut buffer,
                0..,
            )
            .is_err()
        {
            return Err(ProtectionError::DecryptionFailed);
        }
        if cancellation.is_cancelled() {
            return Err(ProtectionError::MaterializationCancelled);
        }
        plaintext.write_all(&buffer)?;
        plaintext_hasher.update(&buffer);
        plaintext_size = plaintext_size
            .checked_add(buffer.len() as u64)
            .ok_or(ProtectionError::InvalidPackage("plaintext size"))?;
    }

    let mut trailing = [0_u8; 1];
    if container.read(&mut trailing)? != 0 {
        return Err(ProtectionError::InvalidPackage("trailing container data"));
    }
    plaintext.flush()?;
    let container_hash: [u8; 32] = container_hasher.finalize().into();
    let plaintext_hash: [u8; 32] = plaintext_hasher.finalize().into();
    if container_size != manifest_file.container_size
        || plaintext_size != manifest_file.plaintext_size
        || container_hash != expected_container_hash
        || plaintext_hash != expected_plaintext_hash
    {
        return Err(ProtectionError::PackageIntegrityMismatch);
    }
    Ok(DecryptResult {
        record_count: manifest_file.record_count,
        plaintext_size,
        plaintext_sha256: plaintext_hash,
    })
}

#[derive(Clone, Copy, Debug)]
struct RecordHeader {
    file_id: u32,
    record_index: u32,
    global_counter: u64,
    plaintext_length: u32,
    ciphertext_length: u32,
}

fn encode_header(
    file_id: u32,
    record_index: u32,
    global_counter: u64,
    plaintext_length: u32,
    ciphertext_length: u32,
) -> [u8; RECORD_HEADER_BYTES] {
    let mut output = [0_u8; RECORD_HEADER_BYTES];
    output[0..8].copy_from_slice(RECORD_MAGIC);
    output[8..10].copy_from_slice(&RECORD_VERSION.to_be_bytes());
    output[10..12].copy_from_slice(&0_u16.to_be_bytes());
    output[12..16].copy_from_slice(&file_id.to_be_bytes());
    output[16..20].copy_from_slice(&record_index.to_be_bytes());
    output[20..28].copy_from_slice(&global_counter.to_be_bytes());
    output[28..32].copy_from_slice(&plaintext_length.to_be_bytes());
    output[32..36].copy_from_slice(&ciphertext_length.to_be_bytes());
    output
}

fn decode_header(bytes: &[u8; RECORD_HEADER_BYTES]) -> Result<RecordHeader> {
    if &bytes[0..8] != RECORD_MAGIC
        || u16::from_be_bytes([bytes[8], bytes[9]]) != RECORD_VERSION
        || u16::from_be_bytes([bytes[10], bytes[11]]) != 0
    {
        return Err(ProtectionError::InvalidPackage("record format"));
    }
    Ok(RecordHeader {
        file_id: u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        record_index: u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
        global_counter: u64::from_be_bytes([
            bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
        ]),
        plaintext_length: u32::from_be_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
        ciphertext_length: u32::from_be_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]),
    })
}

fn encode_nonce(prefix: &[u8; 4], global_counter: u64) -> [u8; 12] {
    let mut nonce = [0_u8; 12];
    nonce[..4].copy_from_slice(prefix);
    nonce[4..].copy_from_slice(&global_counter.to_be_bytes());
    nonce
}

fn encode_aad(
    artifact_id: &[u8; 16],
    file_id: u32,
    record_index: u32,
    global_counter: u64,
    plaintext_length: u32,
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(AAD_DOMAIN.len() + 2 + 16 + 4 + 4 + 8 + 4);
    aad.extend_from_slice(AAD_DOMAIN);
    aad.extend_from_slice(&RECORD_VERSION.to_be_bytes());
    aad.extend_from_slice(artifact_id);
    aad.extend_from_slice(&file_id.to_be_bytes());
    aad.extend_from_slice(&record_index.to_be_bytes());
    aad.extend_from_slice(&global_counter.to_be_bytes());
    aad.extend_from_slice(&plaintext_length.to_be_bytes());
    aad
}

fn read_record<R: Read>(reader: &mut R, buffer: &mut [u8]) -> std::io::Result<usize> {
    let mut offset = 0;
    while offset < buffer.len() {
        match reader.read(&mut buffer[offset..])? {
            0 => break,
            count => offset += count,
        }
    }
    Ok(offset)
}

fn write_hashed<W: Write>(writer: &mut W, hasher: &mut Sha256, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes)?;
    hasher.update(bytes);
    Ok(())
}

fn read_exact_hashed<R: Read>(reader: &mut R, hasher: &mut Sha256, bytes: &mut [u8]) -> Result<()> {
    reader
        .read_exact(bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => {
                ProtectionError::InvalidPackage("truncated container")
            }
            _ => ProtectionError::from(error),
        })?;
    hasher.update(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower_hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        output
    }

    #[test]
    fn round_trip_records_and_reject_tampering() {
        let plaintext = b"0123456789abcdef0123456789";
        let key = [0x42_u8; 32];
        let artifact_id = [0x24_u8; 16];
        let nonce_prefix = [1, 2, 3, 4];
        let mut encrypted = Vec::new();
        let result = encrypt_records(
            plaintext.as_slice(),
            &mut encrypted,
            &key,
            &artifact_id,
            &nonce_prefix,
            7,
            9,
            8,
        )
        .unwrap();
        assert_eq!(result.record_count, 4);
        let golden = decode_lower_hex::<234>(
            "4d50524f54563100000100000000000700000000000000000000000900000008000000083e4603ace302a5536bfb92555b84a6423de77ee37cb53a6e4d50524f54563100000100000000000700000001000000000000000a00000008000000086d3c358bbf037d6d15f6115262cb8f9ef1313bdad45b71404d50524f54563100000100000000000700000002000000000000000b00000008000000089fa71642be0e14ca6e0ee757c3f7eb8fb59328a040456e844d50524f54563100000100000000000700000003000000000000000c0000000200000002c69fcd2fff2fe382baa979f997ec6468c3c0",
            "golden vector",
        )
        .unwrap();
        assert_eq!(encrypted, golden);

        let manifest_file = ProtectedFile {
            file_id: 7,
            container_path: "weights/model.safetensors.protected".to_string(),
            output_path: "model.safetensors".to_string(),
            container_size: result.container_size,
            container_sha256: lower_hex(&result.container_sha256),
            plaintext_size: result.plaintext_size,
            plaintext_sha256: lower_hex(&result.plaintext_sha256),
            record_count: result.record_count,
            first_global_record_counter: 9,
        };
        let mut decrypted = Vec::new();
        decrypt_records(
            encrypted.as_slice(),
            &mut decrypted,
            &key,
            &artifact_id,
            &nonce_prefix,
            &manifest_file,
            8,
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(decrypted, plaintext);

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            decrypt_records(
                golden.as_slice(),
                Vec::new(),
                &key,
                &artifact_id,
                &nonce_prefix,
                &manifest_file,
                8,
                &cancellation,
            ),
            Err(ProtectionError::MaterializationCancelled)
        ));

        encrypted[RECORD_HEADER_BYTES + 1] ^= 1;
        let error = decrypt_records(
            encrypted.as_slice(),
            Vec::new(),
            &key,
            &artifact_id,
            &nonce_prefix,
            &manifest_file,
            8,
            &CancellationToken::default(),
        )
        .unwrap_err();
        assert!(matches!(error, ProtectionError::DecryptionFailed));
    }

    #[test]
    fn rejects_truncated_and_trailing_containers() {
        let key = [3_u8; 32];
        let artifact_id = [4_u8; 16];
        let nonce_prefix = [5_u8; 4];
        let mut encrypted = Vec::new();
        let result = encrypt_records(
            b"protected".as_slice(),
            &mut encrypted,
            &key,
            &artifact_id,
            &nonce_prefix,
            1,
            0,
            32,
        )
        .unwrap();
        let manifest_file = ProtectedFile {
            file_id: 1,
            container_path: "weights/model.safetensors.protected".to_string(),
            output_path: "model.safetensors".to_string(),
            container_size: result.container_size,
            container_sha256: lower_hex(&result.container_sha256),
            plaintext_size: result.plaintext_size,
            plaintext_sha256: lower_hex(&result.plaintext_sha256),
            record_count: result.record_count,
            first_global_record_counter: 0,
        };

        assert!(
            decrypt_records(
                &encrypted[..encrypted.len() - 1],
                Vec::new(),
                &key,
                &artifact_id,
                &nonce_prefix,
                &manifest_file,
                32,
                &CancellationToken::default(),
            )
            .is_err()
        );
        encrypted.push(0);
        assert!(matches!(
            decrypt_records(
                encrypted.as_slice(),
                Vec::new(),
                &key,
                &artifact_id,
                &nonce_prefix,
                &manifest_file,
                32,
                &CancellationToken::default(),
            ),
            Err(ProtectionError::InvalidPackage("trailing container data"))
        ));

        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("synthetic write failure"))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        assert!(matches!(
            decrypt_records(
                &encrypted[..encrypted.len() - 1],
                FailingWriter,
                &key,
                &artifact_id,
                &nonce_prefix,
                &manifest_file,
                32,
                &CancellationToken::default(),
            ),
            Err(ProtectionError::Io)
        ));

        struct FailingReader(bool);
        impl Read for FailingReader {
            fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
                if self.0 {
                    return Err(std::io::Error::other("synthetic read failure"));
                }
                self.0 = true;
                bytes[..3].copy_from_slice(b"key");
                Ok(3)
            }
        }
        assert!(matches!(
            encrypt_records(
                FailingReader(false),
                Vec::new(),
                &key,
                &artifact_id,
                &nonce_prefix,
                1,
                0,
                32,
            ),
            Err(ProtectionError::Io)
        ));
    }
}
