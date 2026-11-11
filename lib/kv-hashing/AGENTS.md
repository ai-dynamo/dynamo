# lib/kv-hashing

`dynamo-kv-hashing` computes the block and sequence hashes engines and the router must agree on, plus the multimodal identity in its `multimodal` module. Changing a hash changes what engines publish and what the router matches, so keep the seed-equality and known-answer tests next to the function.
