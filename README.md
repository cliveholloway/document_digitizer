# Document Digitizer

A Python tool for processing paired scanned handwritten document images into clean, aligned digital documents. Automatically detects rotation, removes scanning artifacts, and stitches overlapping page pairs with precision text preservation.

![Sample Output](docs/sample_before_after.png)

## Features

- **Intelligent Deskewing** - Detects and corrects document rotation using multiple algorithms
- **Artifact Removal** - Eliminates scanning edge artifacts, black bars, and rotation wedges  
- **Precise Stitching** - Aligns overlapping page pairs with template matching
- **Text Preservation** - No-blend stitching maintains crisp, readable text
- **Configurable Processing** - Extensive TOML configuration for different document types
- **Debug Logging** - Detailed processing logs and intermediate file saving

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/document-digitizer.git
cd document-digitizer

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install click opencv-python pillow rich numpy tomli
```

### Basic Usage

1. **Create configuration file:**
```bash
# Copy the sample config
cp config.sample.toml config.toml
```

2. **Organize your scanned images:**
```
input/
├── page001.tif
├── page002.tif  # These will be paired and stitched
├── page003.tif
├── page004.tif  # These will be paired and stitched
└── ...
```

3. **Process your documents:**
```bash
document-digitizer input/ output/
```

4. **Results:**
```
output/
├── pair_001.png  # Processed and stitched pages 1+2
├── pair_002.png  # Processed and stitched pages 3+4
└── ...
```

## Configuration

The tool uses a `config.toml` file for all processing parameters. See [CONFIGURATION.md](CONFIGURATION.md) for detailed configuration options.

### Sample Configuration

```toml
[debug]
save_intermediate = true
log_target = "stdout"

[processing]
angle_threshold = 0.1
max_rotation = 10.0

[cropping]
final_crop_light_threshold = 240
final_crop_dark_threshold = 40

[output]
file_format = "png"
quality = 95
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
6. **Final Crop** - Remove remaining edge artifacts

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

[cropping]
final_crop_dark_threshold = 60  # Catch more age spots/stains
```

### For Poor Quality Scans
```toml
[processing]
overlap_confidence_threshold = 0.15  # More permissive matching

[cropping]
final_crop_background_percent = 0.03  # More aggressive edge removal
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

**Black bars remain** - Edge artifact removal isn't aggressive enough
- Increase `final_crop_dark_threshold` (try 60-80)
- Decrease `final_crop_background_percent` (try 0.03)

**Text is cut off** - Cropping is too aggressive
- Increase `final_crop_background_percent` 
- Check `precrop_max_vertical_percent` isn't too high

See [CONFIGURATION.md](CONFIGURATION.md) for detailed parameter explanations.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Your chosen license here]

## Acknowledgments

Built for digitizing historical handwritten documents with precision and care for text preservation.
