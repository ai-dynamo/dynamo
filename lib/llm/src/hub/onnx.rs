// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read};
use std::path::{Component, Path, PathBuf};

use prost::{Message, encoding::decode_varint};

const MAX_MESSAGE_DEPTH: usize = 100;
const MAX_EXTERNAL_ENTRY_BYTES: u64 = 64 * 1024;

pub(super) fn external_data_complete(path: &Path) -> io::Result<bool> {
    check_file(path, None).map(|complete| complete == Some(true))
}

pub(super) fn shared_external_data_complete(
    path: &Path,
    sidecars: &mut HashSet<PathBuf>,
) -> io::Result<Option<bool>> {
    match path.metadata() {
        Ok(metadata) if !metadata.is_file() => return Ok(None),
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
        Ok(_) => {}
    }
    check_file(path, Some(sidecars))
}

fn check_file(path: &Path, sidecars: Option<&mut HashSet<PathBuf>>) -> io::Result<Option<bool>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Ok(None);
    }
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut reader = BufReader::new(file);
    match check_message(&mut reader, dir, Kind::Model, metadata.len(), 0, sidecars) {
        Ok(complete) => {
            if complete == Some(true) && reader.get_ref().metadata()?.len() != metadata.len() {
                return Ok(None);
            }
            Ok(complete)
        }
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::InvalidData | ErrorKind::UnexpectedEof
            ) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Model,
    Graph,
    Node,
    Attribute,
    Tensor,
    SparseTensor,
    Function,
    TrainingInfo,
}

impl Kind {
    // Fields leading to TensorProto in the ONNX schema (onnx/onnx.proto3).
    fn child(self, field: u64) -> Option<Self> {
        use Kind::*;
        match (self, field) {
            (Model, 7) | (Attribute, 6 | 11) | (TrainingInfo, 1 | 2) => Some(Graph),
            (Model, 20) => Some(TrainingInfo),
            (Model, 25) => Some(Function),
            (Graph, 1) | (Function, 7) => Some(Node),
            (Graph, 5) | (Attribute, 5 | 10) | (SparseTensor, 1 | 2) => Some(Tensor),
            (Graph, 15) | (Attribute, 22 | 23) => Some(SparseTensor),
            (Node, 5) | (Function, 11) => Some(Attribute),
            _ => None,
        }
    }
}

fn check_message(
    reader: &mut BufReader<File>,
    dir: &Path,
    kind: Kind,
    mut remaining: u64,
    depth: usize,
    mut sidecars: Option<&mut HashSet<PathBuf>>,
) -> io::Result<Option<bool>> {
    if depth >= MAX_MESSAGE_DEPTH {
        return Err(invalid_data());
    }
    let mut ir_version = 0;
    let mut graph = false;
    let mut data_location = 0;
    let mut elements = Some(1);
    let mut data_type = 0;
    let mut segment = None;
    let mut external = ExternalData::default();
    let mut complete = Some(true);
    let mut singular_fields = HashSet::new();
    while remaining > 0 {
        let key = read_varint(reader, &mut remaining)?;
        let field = key >> 3;
        if field == 0 || field > u64::from(prost::encoding::MAX_TAG) {
            return Err(invalid_data());
        }
        let wire = key & 7;
        let bytes = match wire {
            0 => {
                let value = read_varint(reader, &mut remaining)?;
                match (kind, field) {
                    (Kind::Model, 1) => ir_version = value,
                    (Kind::Tensor, 1) => add_dimension(&mut elements, value)?,
                    (Kind::Tensor, 2) => data_type = value,
                    (Kind::Tensor, 14) => data_location = value,
                    _ => {}
                }
                continue;
            }
            1 => 8,
            2 => read_varint(reader, &mut remaining)?,
            5 => 4,
            _ => return Err(invalid_data()),
        };
        if depth == 0 && wire == 2 && bytes > remaining {
            return Ok(Some(false));
        }
        remaining = remaining.checked_sub(bytes).ok_or_else(invalid_data)?;

        if wire == 2
            && sidecars.is_some()
            && matches!(
                (kind, field),
                (Kind::Model, 7)
                    | (Kind::TrainingInfo, 1 | 2)
                    | (Kind::Attribute, 5 | 6 | 22)
                    | (Kind::SparseTensor, 1 | 2)
            )
            && !singular_fields.insert(field)
        {
            complete = None;
        }
        if wire == 2
            && let Some(child) = kind.child(field)
        {
            let child_complete = check_message(
                reader,
                dir,
                child,
                bytes,
                depth + 1,
                sidecars.as_deref_mut(),
            )?;
            if child_complete != Some(true) && sidecars.is_none() {
                return Ok(child_complete);
            }
            complete = complete
                .zip(child_complete)
                .map(|(left, right)| left && right);
            graph |= child == Kind::Graph;
        } else if wire == 2 && kind == Kind::Tensor && field == 1 {
            let mut dimensions = bytes;
            while dimensions > 0 {
                add_dimension(&mut elements, read_varint(reader, &mut dimensions)?)?;
            }
        } else if wire == 2 && kind == Kind::Tensor && matches!(field, 3 | 13) {
            if bytes > MAX_EXTERNAL_ENTRY_BYTES {
                return Err(invalid_data());
            }
            let mut data = vec![0; bytes as usize];
            reader.read_exact(&mut data)?;
            if field == 3 {
                segment
                    .get_or_insert_with(TensorSegment::default)
                    .merge(data.as_slice())
                    .map_err(|_| invalid_data())?;
            } else {
                let entry = ExternalEntry::decode(data.as_slice()).map_err(|_| invalid_data())?;
                external.record(entry)?;
            }
        } else {
            // Only inspect graph structure and external references. Inline tensor payloads
            // can be very large and need not be read to check the cache.
            reader.seek_relative(i64::try_from(bytes).map_err(|_| invalid_data())?)?;
        }
    }
    if kind == Kind::Tensor
        && let Some(sidecars) = sidecars
    {
        match external.path(dir) {
            Some(path) => {
                sidecars.insert(path);
            }
            None if data_location == 1 => complete = None,
            None => {}
        }
    }
    let local = match kind {
        Kind::Model => (ir_version > 0 && graph).then_some(true),
        Kind::Tensor if data_location == 1 => {
            match minimum_tensor_bytes(data_type, elements, segment) {
                Some(bytes) => Some(external.complete(dir, bytes)?),
                None => None,
            }
        }
        Kind::Tensor => (data_location == 0 && external.location.is_none()).then_some(true),
        _ => Some(true),
    };
    Ok(complete.zip(local).map(|(left, right)| left && right))
}

fn add_dimension(elements: &mut Option<u128>, dimension: u64) -> io::Result<()> {
    let dimension = i64::try_from(dimension).map_err(|_| invalid_data())?;
    *elements = if dimension == 0 {
        Some(0)
    } else {
        elements.and_then(|count| count.checked_mul(dimension as u128))
    };
    Ok(())
}

#[derive(Message)]
struct TensorSegment {
    #[prost(int64, tag = "1")]
    begin: i64,
    #[prost(int64, tag = "2")]
    end: i64,
}

fn minimum_tensor_bytes(
    data_type: u64,
    elements: Option<u128>,
    segment: Option<TensorSegment>,
) -> Option<u64> {
    let elements = elements?;
    let elements = match segment {
        Some(TensorSegment { begin, end })
            if begin >= 0 && end >= begin && end as u128 <= elements =>
        {
            (end - begin) as u128
        }
        Some(_) => return None,
        None => elements,
    };
    // TensorProto.DataType widths for external/raw data in the ONNX schema.
    let bits = match data_type {
        25 | 26 => 2,
        21..=23 => 4,
        27 | 28 => 6,
        2 | 3 | 9 | 17..=20 | 24 => 8,
        4 | 5 | 10 | 16 => 16,
        1 | 6 | 12 => 32,
        7 | 11 | 13 | 14 => 64,
        15 => 128,
        _ => return None,
    };
    u64::try_from(elements.checked_mul(bits)?.div_ceil(8)).ok()
}

fn read_varint(reader: &mut BufReader<File>, remaining: &mut u64) -> io::Result<u64> {
    let mut bytes = [0; 10];
    for index in 0..bytes.len() {
        *remaining = remaining.checked_sub(1).ok_or_else(invalid_data)?;
        reader.read_exact(&mut bytes[index..=index])?;
        if bytes[index] < 128 {
            return decode_varint(&mut &bytes[..=index]).map_err(|_| invalid_data());
        }
    }
    Err(invalid_data())
}

fn invalid_data() -> io::Error {
    io::Error::new(ErrorKind::InvalidData, "Invalid ONNX model protobuf")
}

#[derive(Message)]
struct ExternalEntry {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

#[derive(Default)]
struct ExternalData {
    location: Option<String>,
    offset: Option<u64>,
    length: Option<u64>,
}

impl ExternalData {
    fn record(&mut self, entry: ExternalEntry) -> io::Result<()> {
        let duplicate = match entry.key.as_str() {
            "location" => self.location.replace(entry.value).is_some(),
            "offset" => self
                .offset
                .replace(entry.value.parse().map_err(|_| invalid_data())?)
                .is_some(),
            "length" => self
                .length
                .replace(entry.value.parse().map_err(|_| invalid_data())?)
                .is_some(),
            _ => false,
        };
        if duplicate {
            return Err(invalid_data());
        }
        Ok(())
    }

    fn path(&self, dir: &Path) -> Option<PathBuf> {
        let location = self.location.as_deref()?;
        let path = Path::new(location);
        (!location.is_empty()
            && path
                .components()
                .all(|part| matches!(part, Component::Normal(_) | Component::CurDir)))
        .then(|| dir.join(path))
    }

    fn complete(&self, dir: &Path, required_bytes: u64) -> io::Result<bool> {
        let Some(path) = self.path(dir) else {
            return Ok(false);
        };
        let offset = self.offset.unwrap_or(0);
        Ok(super::readable_file_len(&path)?
            .and_then(|len| len.checked_sub(offset))
            .is_some_and(|available| {
                self.length
                    .is_none_or(|length| length == 0 || length == required_bytes)
                    && required_bytes <= available
            }))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::hub::stream_model_ready;
    use prost::encoding::{WireType, encode_key, encode_varint};

    fn integer(field: u32, value: u64, body: &mut Vec<u8>) {
        encode_key(field, WireType::Varint, body);
        encode_varint(value, body);
    }

    fn message(field: u32, bytes: &[u8], body: &mut Vec<u8>) {
        encode_key(field, WireType::LengthDelimited, body);
        encode_varint(bytes.len() as u64, body);
        body.extend_from_slice(bytes);
    }

    fn tensor(dim: Option<u8>, external: Option<(&str, &str, Option<&str>)>) -> Vec<u8> {
        let mut body = Vec::new();
        if let Some(dim) = dim {
            message(1, &[dim], &mut body); // packed dims
        }
        integer(2, 1, &mut body); // FLOAT
        message(8, b"weights", &mut body);
        if let Some((location, offset, length)) = external {
            let fields = [
                ("location", Some(location)),
                ("offset", Some(offset)),
                ("length", length),
            ];
            for (key, value) in fields {
                if let Some(value) = value {
                    let entry = ExternalEntry {
                        key: key.into(),
                        value: value.into(),
                    };
                    message(13, &entry.encode_to_vec(), &mut body);
                }
            }
            integer(14, 1, &mut body); // EXTERNAL
        } else {
            message(9, &1_f32.to_le_bytes(), &mut body);
        }
        body
    }

    fn save_model(dir: &Path, graph: &[u8]) -> io::Result<()> {
        let mut model = Vec::new();
        integer(1, 8, &mut model); // ir_version
        message(7, graph, &mut model);
        message(8, &[0x10, 13], &mut model); // default domain, opset 13
        fs::write(dir.join("model.onnx"), model)?;
        fs::write(dir.join("config.json"), b"{}")?;
        fs::write(dir.join("tokenizer.json"), b"{}")?;
        Ok(())
    }

    fn graph_with_output(output_name: &[u8], dim: Option<u8>) -> Vec<u8> {
        let mut graph = Vec::new();
        message(2, b"example", &mut graph);
        let mut output = Vec::new();
        message(1, output_name, &mut output);
        let mut tensor_type = vec![8, 1]; // FLOAT
        let shape = dim.map(|dim| [0x0a, 2, 8, dim]);
        message(
            2,
            shape.as_ref().map_or(&[], |bytes| bytes.as_slice()),
            &mut tensor_type,
        );
        let mut value_type = Vec::new();
        message(1, &tensor_type, &mut value_type);
        message(2, &value_type, &mut output);
        message(12, &output, &mut graph);
        graph
    }

    fn output_graph(initializer: &[u8], dim: Option<u8>) -> Vec<u8> {
        let mut graph = graph_with_output(b"weights", dim);
        message(5, initializer, &mut graph);
        graph
    }

    #[test]
    fn shared_onnx_rejects_top_level_length_overruns() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let mut model = Vec::new();
        integer(1, 8, &mut model);
        message(7, &graph_with_output(b"weights", Some(1)), &mut model);
        model.pop();
        let path = temp.path().join("model.onnx");
        fs::write(&path, model)?;
        assert_eq!(
            shared_external_data_complete(&path, &mut HashSet::new())?,
            Some(false),
        );
        Ok(())
    }

    #[test]
    fn shared_onnx_checks_all_referenced_sidecars() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        let mut graph = output_graph(
            &tensor(Some(1), Some(("./weights.bin", "0", Some("4")))),
            Some(1),
        );
        let mut second = tensor(Some(1), Some(("other.bin", "0", Some("4"))));
        message(8, b"other", &mut second);
        message(5, &second, &mut graph);
        save_model(dir, &graph)?;
        let path = dir.join("model.onnx");
        let mut sidecars = HashSet::new();
        assert_eq!(
            shared_external_data_complete(&path, &mut sidecars)?,
            Some(false)
        );
        assert_eq!(
            sidecars,
            HashSet::from([dir.join("weights.bin"), dir.join("other.bin")])
        );
        fs::write(dir.join("weights.bin"), [0; 4])?;
        fs::write(dir.join("other.bin"), [0; 4])?;
        assert_eq!(
            shared_external_data_complete(&path, &mut sidecars)?,
            Some(true)
        );
        fs::write(dir.join("weights.bin"), [0; 3])?;
        assert_eq!(
            shared_external_data_complete(&path, &mut sidecars)?,
            Some(false)
        );
        Ok(())
    }

    #[test]
    fn shared_onnx_preserves_unknown_and_merged_messages() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        let path = dir.join("model.onnx");
        let mut graph = output_graph(
            &tensor(Some(1), Some(("missing.bin", "0", Some("4")))),
            Some(1),
        );
        let mut unknown = tensor(Some(1), Some(("future.bin", "0", Some("4"))));
        integer(2, 99, &mut unknown);
        message(8, b"future", &mut unknown);
        message(5, &unknown, &mut graph);
        save_model(dir, &graph)?;
        assert!(!external_data_complete(&path)?);
        let mut sidecars = HashSet::new();
        assert_eq!(shared_external_data_complete(&path, &mut sidecars)?, None);
        assert_eq!(
            sidecars,
            HashSet::from([dir.join("missing.bin"), dir.join("future.bin")])
        );

        fs::write(dir.join("weights.bin"), [0; 4])?;
        let initializer = tensor(Some(1), Some(("weights.bin", "0", Some("4"))));
        save_model(dir, &output_graph(&initializer, Some(1)))?;
        let mut repeated_graph = Vec::new();
        message(7, &[], &mut repeated_graph);
        repeated_graph.extend_from_slice(&fs::read(&path)?);
        fs::write(&path, repeated_graph)?;
        assert!(external_data_complete(&path)?);
        assert_eq!(shared_external_data_complete(&path, &mut sidecars)?, None);

        let mut attribute = Vec::new();
        message(1, b"value", &mut attribute);
        integer(20, 4, &mut attribute);
        let mut descriptor = Vec::new();
        message(1, &[1], &mut descriptor);
        integer(2, 1, &mut descriptor);
        integer(14, 1, &mut descriptor);
        let mut data = Vec::new();
        message(
            13,
            &ExternalEntry {
                key: "location".into(),
                value: "weights.bin".into(),
            }
            .encode_to_vec(),
            &mut data,
        );
        message(5, &descriptor, &mut attribute);
        message(5, &data, &mut attribute);
        let mut node = Vec::new();
        message(2, b"weights", &mut node);
        message(4, b"Constant", &mut node);
        message(5, &attribute, &mut node);
        let mut graph = graph_with_output(b"weights", Some(1));
        message(1, &node, &mut graph);
        save_model(dir, &graph)?;
        assert!(!external_data_complete(&path)?);
        assert_eq!(shared_external_data_complete(&path, &mut sidecars)?, None);
        assert!(sidecars.contains(&dir.join("weights.bin")));
        Ok(())
    }

    #[test]
    fn shared_onnx_preserves_io_errors_and_strict_early_exit() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        fs::write(dir.join("not-a-directory"), b"file")?;
        let mut graph = output_graph(
            &tensor(Some(1), Some(("missing.bin", "0", Some("4")))),
            Some(1),
        );
        message(
            5,
            &tensor(
                Some(1),
                Some(("not-a-directory/weights.bin", "0", Some("4"))),
            ),
            &mut graph,
        );
        save_model(dir, &graph)?;
        let path = dir.join("model.onnx");
        assert!(!external_data_complete(&path)?);
        assert_eq!(
            shared_external_data_complete(&path, &mut HashSet::new())
                .unwrap_err()
                .kind(),
            ErrorKind::NotADirectory
        );
        fs::write(dir.join("missing.bin"), [0; 4])?;
        assert_eq!(
            external_data_complete(&path).unwrap_err().kind(),
            ErrorKind::NotADirectory
        );
        Ok(())
    }

    #[test]
    fn onnx_cache_requires_external_tensor_data() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        save_model(dir, &output_graph(&tensor(Some(1), None), Some(1)))?;
        assert!(stream_model_ready(dir, false)?);

        save_model(
            dir,
            &output_graph(
                &tensor(Some(1), Some(("./weights.bin", "3", Some("4")))),
                Some(1),
            ),
        )?;
        assert!(stream_model_ready(dir, true)?);
        assert!(
            !stream_model_ready(dir, false)?,
            "missing external tensor data"
        );
        let data = dir.join("weights.bin");
        fs::write(&data, [0; 6])?;
        assert!(
            !stream_model_ready(dir, false)?,
            "truncated external tensor data"
        );
        fs::write(&data, [0; 7])?;
        assert!(stream_model_ready(dir, false)?);

        save_model(
            dir,
            &output_graph(
                &tensor(
                    Some(1),
                    Some(("weights.bin", "18446744073709551615", Some("4"))),
                ),
                Some(1),
            ),
        )?;
        assert!(!stream_model_ready(dir, false)?, "overflowing tensor range");
        Ok(())
    }

    #[test]
    fn onnx_cache_distinguishes_empty_tensor_from_missing_bytes() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        let data = dir.join("weights.bin");
        fs::write(&data, [])?;
        for dim in [None, Some(0)] {
            let initializer = tensor(dim, Some(("weights.bin", "0", None)));
            save_model(dir, &output_graph(&initializer, dim))?;
            assert_eq!(stream_model_ready(dir, false)?, dim == Some(0));
        }
        for length in [None, Some("0")] {
            let initializer = tensor(None, Some(("weights.bin", "4", length)));
            save_model(dir, &output_graph(&initializer, None))?;
            fs::write(&data, [0; 3])?;
            assert!(!stream_model_ready(dir, false)?, "offset past EOF");
            fs::write(&data, [0; 4])?;
            assert!(!stream_model_ready(dir, false)?, "offset at EOF");
            fs::write(&data, [0; 7])?;
            assert!(
                !stream_model_ready(dir, false)?,
                "three bytes for a FLOAT scalar"
            );
            fs::write(&data, [0; 8])?;
            assert!(stream_model_ready(dir, false)?);
        }

        let initializer = tensor(Some(1), Some(("weights.bin", "4", Some("3"))));
        save_model(dir, &output_graph(&initializer, Some(1)))?;
        assert!(
            !stream_model_ready(dir, false)?,
            "declared length shorter than the tensor"
        );
        Ok(())
    }

    #[test]
    fn onnx_cache_checks_exact_declared_tensor_length() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        fs::write(dir.join("weights.bin"), [0; 12])?;
        for (dim, length, expected) in [
            (1, None, true),
            (1, Some("3"), false),
            (1, Some("4"), true),
            (1, Some("5"), false),
            (1, Some("0"), true),
            (0, Some("0"), true),
        ] {
            let initializer = tensor(Some(dim), Some(("weights.bin", "3", length)));
            save_model(dir, &output_graph(&initializer, Some(dim)))?;
            assert_eq!(
                stream_model_ready(dir, false)?,
                expected,
                "dim={dim}, length={length:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn onnx_cache_merges_repeated_tensor_segments() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        let weights = dir.join("weights.bin");
        let initializer = tensor(Some(8), Some(("weights.bin", "0", None)));

        // Both encodings merge to segment [2, 3), which requires one FLOAT.
        fs::write(&weights, [0; 4])?;
        for parts in [[[8, 2], [16, 3]], [[16, 3], [8, 2]]] {
            let mut tensor = initializer.clone();
            for part in parts {
                message(3, &part, &mut tensor);
            }
            save_model(dir, &output_graph(&tensor, Some(8)))?;
            assert!(stream_model_ready(dir, false)?, "merge {parts:?}");
        }

        // A later explicit zero overrides begin and preserves the earlier end.
        let mut tensor = initializer;
        message(3, &[8, 2, 16, 3], &mut tensor);
        message(3, &[8, 0], &mut tensor);
        save_model(dir, &output_graph(&tensor, Some(8)))?;
        assert!(!stream_model_ready(dir, false)?);
        fs::write(&weights, [0; 12])?;
        assert!(stream_model_ready(dir, false)?);
        Ok(())
    }

    #[test]
    fn onnx_external_segment_defaults_to_zero_start() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        let mut initializer = tensor(Some(8), Some(("weights.bin", "0", None)));
        message(3, &[0x10, 1], &mut initializer); // segment [0, 1), only end is encoded
        save_model(dir, &output_graph(&initializer, Some(8)))?;
        fs::write(dir.join("weights.bin"), [0; 4])?;
        assert!(stream_model_ready(dir, false)?);
        fs::write(dir.join("weights.bin"), [0; 3])?;
        assert!(!stream_model_ready(dir, false)?);
        Ok(())
    }

    #[test]
    fn onnx_packed_tensor_size_rounds_up() {
        assert_eq!(minimum_tensor_bytes(21, Some(3), None), Some(2)); // UINT4
        assert_eq!(minimum_tensor_bytes(25, Some(5), None), Some(2)); // UINT2
        assert_eq!(minimum_tensor_bytes(27, Some(5), None), Some(4)); // FLOAT6
        assert_eq!(minimum_tensor_bytes(1, Some(u128::MAX), None), None);
        assert_eq!(minimum_tensor_bytes(0, Some(1), None), None);
        assert_eq!(
            minimum_tensor_bytes(1, Some(8), Some(TensorSegment { begin: 2, end: 3 })),
            Some(4),
        );
    }

    #[test]
    fn onnx_cache_checks_sparse_tensors_in_attribute_subgraphs() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let dir = temp.path();
        let mut indices = Vec::new();
        integer(1, 1, &mut indices); // dims = [1]
        integer(2, 7, &mut indices); // INT64
        message(9, &0_i64.to_le_bytes(), &mut indices);
        let mut sparse = Vec::new();
        message(
            1,
            &tensor(Some(1), Some(("weights.bin", "0", Some("4")))),
            &mut sparse,
        );
        message(2, &indices, &mut sparse);
        integer(3, 1, &mut sparse);

        let mut then_graph = graph_with_output(b"weights", Some(1));
        message(15, &sparse, &mut then_graph); // GraphProto.sparse_initializer
        let mut node = Vec::new();
        message(1, b"condition", &mut node);
        message(2, b"result", &mut node);
        message(4, b"If", &mut node);
        for (name, branch) in [
            (b"then_branch".as_slice(), then_graph),
            (
                b"else_branch".as_slice(),
                output_graph(&tensor(Some(1), None), Some(1)),
            ),
        ] {
            let mut attribute = Vec::new();
            message(1, name, &mut attribute);
            message(6, &branch, &mut attribute); // AttributeProto.g
            integer(20, 5, &mut attribute); // GRAPH
            message(5, &attribute, &mut node); // NodeProto.attribute
        }
        let mut condition = Vec::new();
        integer(2, 9, &mut condition); // scalar BOOL
        message(8, b"condition", &mut condition);
        message(9, &[1], &mut condition);
        let mut graph = graph_with_output(b"result", Some(1));
        message(1, &node, &mut graph); // GraphProto.node
        message(5, &condition, &mut graph); // GraphProto.initializer
        save_model(dir, &graph)?;
        assert!(!stream_model_ready(dir, false)?);
        fs::write(dir.join("weights.bin"), [0; 4])?;
        assert!(stream_model_ready(dir, false)?);
        Ok(())
    }
}
