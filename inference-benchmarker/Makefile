build:
	cargo build --release --package inference-benchmarker --bin inference-benchmarker

run: build
	cargo run --package inference-benchmarker --bin inference-benchmarker -- $@

test:
	cargo test --package inference-benchmarker

lint:
	cargo +nightly fmt
	cargo clippy --package inference-benchmarker
