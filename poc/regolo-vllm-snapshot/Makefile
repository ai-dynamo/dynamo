.PHONY: test verify-v0-checksum verify-implementation-checksum syntax plan

test: verify-v0-checksum
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s verification/v0/tests -v
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s implementation/tests -v

verify-v0-checksum:
	sha256sum --check verification/v0/SHA256SUMS

verify-implementation-checksum:
	sha256sum --check implementation/SHA256SUMS

syntax:
	python3 -c 'import ast,pathlib; [ast.parse(p.read_text(), filename=str(p)) for p in pathlib.Path(".").rglob("*.py") if "__pycache__" not in p.parts]'
	bash -n implementation/bin/build-placeholder implementation/bin/build-snapshotctl implementation/bin/install-snapshot-chart implementation/bin/cleanup-namespace implementation/image/snapshot-entrypoint

plan:
	implementation/bin/generate-run-plan --output-dir artifacts/v1
