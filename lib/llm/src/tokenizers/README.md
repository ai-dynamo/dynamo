# Tokenizers

## Introduction
`tokenizers` is designed for efficient and versatile tokenization in natural language processing. It supports both HuggingFace models, offering a streamlined API for text encoding and decoding.

## Features
- **Hash Verification**: Ensures tokenization consistency and accuracy across different models.
- **Simple Encoding and Decoding**: Facilitates the conversion of text to token IDs and back.
- **Sequence Management**: Manage sequences of tokens for complex NLP tasks effectively.

## Quick Start

#### HuggingFace Tokenizer
```rust
use dynamo_llm::tokenizers::hf::HuggingFaceTokenizer;

let hf_tokenizer = HuggingFaceTokenizer::from_file("tests/data/sample-models/TinyLlama_v1.1/tokenizer.json")
    .expect("Failed to load HuggingFace tokenizer");
```

### Encoding and Decoding Text

```rust
use dynamo_llm::tokenizers::{HuggingFaceTokenizer, traits::{Encoder, Decoder}};

let tokenizer = HuggingFaceTokenizer::from_file("tests/data/sample-models/TinyLlama_v1.1/tokenizer.json")
    .expect("Failed to load HuggingFace tokenizer");

let text = "Your sample text here";
let encoding = tokenizer.encode(text)
    .expect("Failed to encode text");

println!("Encoding: {:?}", encoding);

let decoded_text = tokenizer.decode(&encoding.token_ids, false)
    .expect("Failed to decode token IDs");

assert_eq!(text, decoded_text);

// Using the Sequence object for encoding and decoding

use dynamo_llm::tokenizers::{Sequence, Tokenizer};
use std::sync::{Arc, RwLock};

let tokenizer = Tokenizer::from(Arc::new(tokenizer));
let mut sequence = Sequence::new(tokenizer.clone());

sequence.append_text("Your sample text here")
    .expect("Failed to append text");

let delta = sequence.append_token_id(1337)
    .expect("Failed to append token_id");
```
