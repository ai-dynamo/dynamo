cargo build
cd lib/bindings/python
maturin develop --uv
cd ../../../
uv pip install .
