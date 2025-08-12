# Document Digitizer

A Python tool for processing paired, scanned, handwritten document images into clean, aligned digital documents. Automatically detects rotation, removes scanning artifacts, and stitches overlapping page pairs with precision text preservation.

![Example](docs/example.md)

## Features

- **Intelligent Deskewing** - Detects and corrects document rotation using multiple algorithms
- **Artifact Removal** - Eliminates edge artifacts and rotation wedges  
- **Precise Stitching** - Aligns overlapping page pairs with template matching
- **Text Preservation** - No-blend stitching maintains crisp, readable text
- **Configurable Processing** - Extensive TOML configuration for different quality documents
- **Debug Logging** - Detailed processing logs and intermediate file saving

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/cliveholloway/document_digitizer.git
cd document_digitizer

# Create and activate virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install the package and dependencies
pip install -e .```

### Basic Usage

1. **Create configuration file:**
```bash
# Copy the sample config
cp config.sample.toml config.toml
```

For your initial testing you can probably leave it as-is, but when running a project,
I suggest you change the debug options:

```toml
[debug]
# Save intermediate processing images (a_precrop, b_rotated, c_deskewed, d_stitched)
save_intermediate = true

# Logging target: "stdout" for console output, "file" for log file + minimal console
log_target = "stdout"
```
Assuming the final images are fine, you can set `save_inermediate` to false, but it
will give you a useful idea as to what the steps are, and be helpful in debugging
problematic input pairs.

2. **Organize your scanned images:**

They will be processed in pairs, in alphabetical order.

```
input/
├── image001.tif
├── image002.tif  # These will be paired and stitched
├── image003.tif
├── image004.tif  # These will be paired and stitched
└── ...
```

3. **Process your documents:**
```bash
document-digitizer input/ output/
```

4. **Results:**
```
output/
├── pair_001.png  # Processed and stitched images 1+2
├── pair_002.png  # Processed and stitched images 3+4
└── ...
```

## Configuration

The tool uses a `config.toml` file for all processing parameters. See [docs/configuration.md](configuration.md) for detailed configuration options.

### Sample Configuration

```toml
[debug]
save_intermediate = true
log_target = "stdout"

[processing]
angle_threshold = 0.1
max_rotation = 10.0

[cropping]
precrop_max_vertical_percent = 10

[output]
file_format = "png"
```

## Advanced Usage

### Custom Configuration
```bash
document-digitizer input/ output/ --config custom_config.toml
```

### File Logging Mode
Set `log_target = "file"` in config for minimal console output with detailed log files.

### Different Output Formats
```toml
[output]
file_format = "jpg"  # or "png", "tiff"
quality = 95         # for JPG only
```

## Processing Pipeline

1. **Pre-crop** - Remove manual rotation artifacts from scanning
2. **Deskew** - Detect and correct text line rotation (±10°)
3. **Math Crop** - Remove rotation wedges using trigonometry
4. **Template Match** - Find optimal overlap between page pairs
5. **Stitch** - Combine images with hard cut (no blending)
6. **Final Crop** - Crop edges to remove X-axis offset, if possible.

## Debug Output

With `save_intermediate = true`, the tool saves processing stages:

```
output/
├── pair_001_a_page1_precrop.png    # After pre-processing
├── pair_001_b_page1_rotated.png    # After rotation (with wedges)
├── pair_001_c_page1_deskewed.png   # After math crop (clean)
├── pair_001_d_stitched.png         # After stitching
└── pair_001.png                    # Final result
```

## Configuration Tuning

### For Old/Fragile Documents
```toml
[processing]
angle_threshold = 0.05  # More sensitive rotation detection
max_rotation = 5.0      # Conservative rotation limit
```

### For Poor Quality Scans
```toml
[processing]
overlap_confidence_threshold = 0.15  # More permissive matching
```

### For High Quality Documents
```toml
[output]
file_format = "tiff"  # Lossless archival format
```

## Requirements

- Python 3.9+
- OpenCV 4.8+
- NumPy 1.24+
- Rich 13.0+ (for console output)
- Click 8.0+ (for CLI)
- tomli (for Python <3.11) or built-in tomllib

## Troubleshooting

### Common Issues

**"No overlap found"** - Images don't have sufficient matching content
- Check that pages are sequential and overlapping
- Reduce `overlap_confidence_threshold` in config

**"Angle seems too large"** - Rotation detection is finding false positives  
- Reduce `max_rotation` to be more conservative
- Check that images aren't severely rotated (>10°)

**Text is cut off** - Cropping is too aggressive
- Check `precrop_max_vertical_percent` isn't too high

See [docs/configuration.md](configuration.md) for detailed parameter explanations.

## Contributing

Contributions welcome! Please read [docs/contributing.md](contributing.md) for guidelines.

## License

[LICENSE](LICENSE)

## Acknowledgments

Claude did the heavy lifting.
