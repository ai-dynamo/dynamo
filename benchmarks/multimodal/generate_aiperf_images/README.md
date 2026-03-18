# Generate aiperf Source Images

aiperf's built-in image generator ships with very few source images. When
benchmarking with `--image-mode base64`, aiperf picks from its
`assets/source_images/` directory — a small set means every request sends
nearly identical images, which doesn't stress the multimodal pipeline
realistically.

This script populates that directory with 200 random-noise PNGs so aiperf
has a larger pool to sample from.

## Usage

```bash
python main.py
```

Images are written directly into aiperf's installed `source_images/` directory.
