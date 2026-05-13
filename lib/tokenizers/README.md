# dynamo-tokenizers

Efficient, versatile tokenization for LLM inference. Wraps HuggingFace and TikToken tokenizers (plus a FastTokenizer hybrid mode) behind a small encode/decode/sequence API designed for streaming detokenization.

## Features

- **Multiple backends.** HuggingFace `tokenizers`, OpenAI `tiktoken`, and a FastTokenizer hybrid behind one trait.
- **Streaming-friendly.** `Sequence` tracks incremental token-id appends and emits text deltas without re-decoding the full prefix.
- **Hash verification.** Detect tokenizer drift across model versions.

## Quick start

```rust
use dynamo_tokenizers::hf::HuggingFaceTokenizer;
use dynamo_tokenizers::traits::{Encoder, Decoder};

// tokenizer.json downloaded from any HuggingFace model repo
let tokenizer = HuggingFaceTokenizer::from_file("/path/to/tokenizer.json")
    .expect("load tokenizer");

let encoding = tokenizer.encode("Your sample text here")
    .expect("encode");
println!("{:?}", encoding);

let decoded = tokenizer.decode(&encoding.token_ids, false)
    .expect("decode");
assert_eq!(decoded, "Your sample text here");
```

## Streaming detokenization with `Sequence`

```rust
use dynamo_tokenizers::{Sequence, Tokenizer};
use std::sync::Arc;

let tokenizer = Tokenizer::from(Arc::new(tokenizer));
let mut sequence = Sequence::new(tokenizer.clone());

sequence.append_text("Your sample text here")
    .expect("append text");

// As each new token id is produced by the engine, append it
// and get back just the incremental text delta:
let delta = sequence.append_token_id(1337)
    .expect("append token_id");
```
