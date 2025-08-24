# Document Digitizer

A Python toolkit for processing scanned documents, featuring automatic deskewing and intelligent image stitching capabilities.

## Features

- **Deskewing**: Automatically detects and corrects document rotation using advanced Hough line detection
- **Image Stitching**: Intelligently combines paired document scans with overlap detection
- **Multiple Formats**: Supports TIFF, PNG, JPEG, BMP, WebP, and JPEG 2000 input formats
- **Flexible Logging**: Configurable console or file-based logging with verbose debugging
- **Robust Processing**: Timeout protection and fallback mechanisms for reliable batch processing

## Quick Start

```bash
# Clone and set up virtual environment
git clone https://github.com/cliveholloway/document_digitizer.git
cd document_digitizer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Copy and edit (if needed) the configuration file
cp config.sample.toml config.toml

# Deskew scanned images
document-deskew input_dir output_dir -v

# Stitch pairs of deskewed document images  
document-stitch input_dir output_dir -v
```

## Installation

### Requirements
- Python 3.9+
- OpenCV
- NumPy  
- Click
- Rich
- Pillow
- tomli (for Python < 3.11)

### Install from Source
```bash
git clone https://github.com/cliveholloway/document_digitizer.git
cd document_digitizer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

## Usage

This code has two steps that batch process images:

- deskew - straighten the original scans and crop wedges after
- stitch - join together 2 images, one above the other

### Deskewing Documents

The deskew tool automatically detects and corrects document rotation:

```bash
# Basic usage - using the installed script
document-deskew input_folder output_folder

# With verbose logging
document-deskew input_folder output_folder -v

# With custom configuration  
document-deskew input_folder output_folder --config my_config.toml
```

**Input formats supported**: TIFF, PNG, JPEG, BMP, WebP, JPEG 2000  
**Output format**: PNG (high quality compression)

### Stitching Document Pairs

The stitch tool combines paired document scans with intelligent overlap detection:

```bash
# Stitch pairs of PNG images
document-stitch input_folder output_folder -v

# Images are processed in alphabetical pairs:
# image001.png + image002.png → page001.png
# image003.png + image004.png → page002.png
```

Note, naming images in pairs starting at 001 will ensure the stitched image
reflects the source, making identifying problematic (failed) stitch source
images. ie:
```bash
# image001a.png + image001b.png → page001.png
# image002a.png + image002b.png → page002.png
```

**Input format**: PNG files (pairs processed alphabetically)  
**Output format**: PNG files named `page001.png`, `page002.png`, etc.

## Configuration

Copy `config.sample.toml` to `config.toml` and adjust settings if needed:

```toml
[debug]
# "stdout" for console output, "file" for detailed log files
log_target = "stdout"

[deskew]
# Minimum angle to trigger rotation (degrees)
angle_threshold = 0.1

# Maximum angle to attempt (degrees)
max_rotation = 10.0

[stitching]
# Maximum overlap search area (pixels)
max_y_overlap = 1500

# Maximum horizontal offset (pixels)
max_x_offset = 300

# Confidence thresholds (0.0-1.0)
overlap_confidence_threshold = 0.6
overlap_confidence_backup = 0.4
```

See [Configuration Guide](docs/configuration.md) for detailed explanations.

## How It Works

### Deskewing Algorithm

1. **Multi-Method Analysis**: Uses probabilistic and standard Hough line detection plus projection analysis
2. **Robust Detection**: Combines results from multiple algorithms for reliable angle detection
3. **Smart Filtering**: Rejects obviously incorrect results and handles edge cases
4. **Precise Rotation**: Applies mathematical rotation with intelligent cropping to remove artifacts

### Stitching Algorithm

1. **Template Matching**: Extracts a sample from one image and searches for it in the other
2. **Overlap Detection**: Calculates precise Y-overlap and X-offset between images
3. **Confidence Scoring**: Uses correlation coefficients to validate matches
4. **Fallback Handling**: Falls back to simple concatenation when overlap detection fails

## Sample Data

Sample scanned documents are included in `data/sample_scans/` for testing. These are
the first 2 of around 700 scans that inspired this project.

If you want to know more, you can [read the whole journal](https://HerbertHolloway.org)

```bash
# Test deskewing with samples
document-deskew data/sample_scans output/deskewed -v

# and stitching
document-stitch output/deskewed output/stitched -v
```

## Documentation

- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [Configuration Guide](docs/configuration.md) - Complete configuration reference
- [Usage Examples](docs/example.md) - Common workflows and examples
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Logging

Both tools support flexible logging:

- **Console Mode** (`log_target = "stdout"`): Progress messages to console, verbose details with `-v`
- **File Mode** (`log_target = "file"`): Complete logs saved to files, console shows progress only

Log files are saved as `deskew.log` and `stitch.log` in the output directory.

## Performance Tips

- **Batch Processing**: Both tools are optimized for processing many files efficiently
- **Timeout Protection**: Deskewing includes timeout mechanisms to prevent hanging on difficult images
- **Memory Efficiency**: Images are processed one at a time to minimize memory usage
- **Format Optimization**: Output PNG files use high-quality compression for good file sizes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE file

## About this project

After 30 years of coding mainly in Perl, I'm teaching myself Python and coding with LLMs.
This is the generic first part of the project.  The rest is only relevant for my project, so the
AI code will be in [this repo](https://github.com/cliveholloway/herbert/) soon.

And yes, I know I haven't included tests. I'll get around to that when I'm more familiar with Python.

I have also not closely analyzed the vibe coded parts of this, so apologies if not the most efficient!

cLive ;-) <clive.holloway@gmail.com>
