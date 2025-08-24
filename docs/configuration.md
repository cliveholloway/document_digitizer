# Configuration Guide

The Document Digitizer uses TOML configuration files to control processing behavior. This guide explains all available settings.

## Configuration File Setup

1. **Copy the sample configuration:**
   ```bash
   cp config.sample.toml config.toml
   ```

2. **Edit the settings** for your needs

3. **Use with tools:**
   ```bash
   # Uses config.toml by default
   document-deskew input output
   
   # Use custom config file
   document-deskew input output --config my_settings.toml
   ```

## Configuration Sections

### [debug] - Logging Configuration

Controls how the tools output information during processing.

```toml
[debug]
log_target = "stdout"
```

**Options:**
- `"stdout"`: Log to console. Shows progress messages normally, detailed debug info only with `-v` flag
- `"file"`: Log to files. Saves complete logs to `deskew.log`/`stitch.log` in output directory, shows minimal progress on console

**When to use:**
- **stdout**: For interactive use, testing, or when you want to see everything in the terminal
- **file**: For batch processing, automation, or when you need complete logs for troubleshooting

### [deskew] - Document Rotation Correction

Controls the automatic deskewing behavior for correcting document rotation.

```toml
[deskew]
angle_threshold = 0.1
max_rotation = 10.0
```

#### angle_threshold
**Type:** Float (degrees)  
**Default:** 0.1  
**Range:** 0.01 - 5.0

Minimum rotation angle required to trigger correction. Images with detected rotation below this threshold are considered "straight enough" and won't be rotated.

**Examples:**
- `0.05`: Very sensitive - corrects tiny rotations
- `0.1`: Default - good balance of correction vs avoiding unnecessary processing  
- `0.5`: Less sensitive - only corrects noticeable rotations
- `1.0`: Conservative - only corrects obvious rotations

**Usage tips:**
- Lower values: More corrections, but may "fix" images that don't need it
- Higher values: Fewer corrections, may leave slightly skewed images uncorrected

#### max_rotation
**Type:** Float (degrees)  
**Default:** 10.0  
**Range:** 1.0 - 45.0

Maximum rotation angle that the algorithm will attempt to correct. Larger rotations are likely detection errors and will be skipped to avoid damaging the document.

**Examples:**
- `5.0`: Conservative - only fixes slight skews
- `10.0`: Default - handles typical scanner skew
- `15.0`: Aggressive - handles more severe rotations
- `30.0`: Very aggressive - may attempt to fix severely rotated scans

**Usage tips:**
- Typical scanner skew: 0.1° - 5°
- Severely skewed documents: up to 15°
- Values above 20° risk false positives from text/image content

### [stitching] - Image Combination Settings

Controls how paired document images are combined with overlap detection.

```toml
[stitching]
max_y_overlap = 1500
max_x_offset = 300
overlap_confidence_threshold = 0.6
overlap_confidence_backup = 0.4
```

#### max_y_overlap
**Type:** Integer (pixels)  
**Default:** 1500  
**Range:** 100 - 5000

Maximum vertical overlap area to search when detecting where images connect. Larger values allow detection of bigger overlaps but increase processing time.

**Examples:**
- `500`: Fast processing, handles small overlaps (0.5-1 inch at 300 DPI)
- `1500`: Default - handles typical book page overlaps (1.5 inches at 300 DPI)
- `3000`: Handles large overlaps but slower processing (3 inches at 300 DPI)

**Usage tips:**
- Match your scanning overlap: if you overlap 1 inch, use ~300-500 pixels
- Larger values = more thorough search but slower processing
- Too small = may miss actual overlaps, falling back to simple concatenation

#### max_x_offset
**Type:** Integer (pixels)  
**Default:** 300  
**Range:** 50 - 1000

Maximum horizontal misalignment allowed between paired images. Accounts for slight positioning differences when scanning.

**Examples:**
- `100`: Strict alignment - requires very consistent scanning
- `300`: Default - tolerates typical handheld scanning variations
- `500`: Loose alignment - handles inconsistent positioning

**Usage tips:**
- Flatbed scanners: can use lower values (100-200)
- Handheld scanners: need higher values (300-500)
- Too strict = rejects valid matches
- Too loose = accepts false matches

#### overlap_confidence_threshold
**Type:** Float (correlation coefficient)  
**Default:** 0.6  
**Range:** 0.1 - 0.95

Primary confidence threshold for accepting overlap detection. Higher values require better matches but reduce false positives.

**Examples:**
- `0.4`: Lenient - accepts lower quality matches
- `0.6`: Default - good balance of accuracy vs acceptance
- `0.8`: Strict - only accepts high-confidence matches

**Technical details:**
- Uses normalized cross-correlation (0.0 = no match, 1.0 = perfect match)
- Values above 0.9 are rarely achieved with real scanned documents
- Lower values may accept false matches from repeated patterns

#### overlap_confidence_backup
**Type:** Float (correlation coefficient)  
**Default:** 0.4  
**Range:** 0.1 - 0.8

Fallback confidence threshold used when primary threshold fails. Provides a second chance for lower-quality but potentially valid matches.

**Examples:**
- `0.3`: Very lenient fallback
- `0.4`: Default - catches medium-quality matches the primary missed
- `0.5`: Conservative fallback - still maintains quality

**Usage tips:**
- Should be lower than `overlap_confidence_threshold`
- Helps with challenging documents (poor scan quality, complex layouts)
- Set too low = more false positives
- Set too high = may as well use single threshold

## Configuration Examples

### High-Quality Scans (Flatbed Scanner)
```toml
[debug]
log_target = "stdout"

[deskew]
angle_threshold = 0.05  # Very precise
max_rotation = 5.0      # Conservative limit

[stitching]
max_y_overlap = 1000    # Consistent overlaps
max_x_offset = 150      # Precise alignment
overlap_confidence_threshold = 0.7  # High quality
overlap_confidence_backup = 0.5
```

### Mobile/Handheld Scanning
```toml
[debug]
log_target = "file"  # Save logs for review

[deskew]
angle_threshold = 0.2   # Less sensitive
max_rotation = 15.0     # Handle more skew

[stitching]
max_y_overlap = 2000    # Variable overlaps
max_x_offset = 500      # Loose alignment
overlap_confidence_threshold = 0.5  # Lower quality
overlap_confidence_backup = 0.35
```

### Batch Processing
```toml
[debug]
log_target = "file"  # Complete logs

[deskew]
angle_threshold = 0.1
max_rotation = 10.0

[stitching]
max_y_overlap = 1500
max_x_offset = 300
overlap_confidence_threshold = 0.6
overlap_confidence_backup = 0.4
```

### Troubleshooting/Debug Mode
```toml
[debug]
log_target = "stdout"  # See everything in console

[deskew]
angle_threshold = 0.05  # Catch small rotations
max_rotation = 20.0     # Don't skip potential issues

[stitching]
max_y_overlap = 2500    # Thorough search
max_x_offset = 400      # Accept more variation
overlap_confidence_threshold = 0.5  # Lower bar
overlap_confidence_backup = 0.3     # Very permissive
```

## Performance vs Quality Trade-offs

### Faster Processing
- Increase `angle_threshold` (fewer rotations processed)
- Decrease `max_y_overlap` (smaller search areas)
- Decrease `max_x_offset` (less alignment checking)
- Use `log_target = "file"` (less console output)

### Higher Quality Results
- Decrease `angle_threshold` (catch subtle rotations)
- Increase `max_y_overlap` (more thorough overlap search)
- Increase `overlap_confidence_threshold` (stricter matching)
- Use `log_target = "stdout"` with `-v` flag (see what's happening)

## Validation and Testing

Test your configuration with sample documents:

```bash
# Test deskewing with different thresholds
document-deskew sample_scans test_output -v

# Test stitching with different confidence levels
python -m document_digitizer.stitch sample_pairs test_output -v
```

Monitor the logs to see:
- How many images are being rotated vs skipped
- Confidence scores of successful/failed stitches
- Processing times per image

Adjust settings based on results and your document types.
