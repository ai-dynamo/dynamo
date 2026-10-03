# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate tokenizer.json — the bundled byte-level mock tokenizer.

BPE with an empty merges list degenerates to one token per byte: the
ByteLevel pre-tokenizer maps every byte to a printable character, those
256 characters are the whole vocabulary, and with no merges nothing
ever combines. Any text tokenizes, distinct texts stay distinct, and
decode is byte-exact. <|endoftext|> (id 256) is the EOS token — outside
the byte range so it can never collide with real text.

Run from this directory:  python generate.py
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers


def main() -> None:
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|endoftext|>"])
    tok.save("tokenizer.json")

    # sanity: byte-exact roundtrip and stable EOS id
    check = Tokenizer.from_file("tokenizer.json")
    sample = "Hello! I am a hand-written Dynamo engine."
    if check.decode(check.encode(sample).ids) != sample:
        raise RuntimeError("byte-level roundtrip is not exact")
    if check.token_to_id("<|endoftext|>") != 256:
        raise RuntimeError("EOS token id is not 256")
    print("tokenizer.json regenerated and verified")


if __name__ == "__main__":
    main()
