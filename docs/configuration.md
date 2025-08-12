# Configuration Guide

The Document Digitizer uses a comprehensive TOML configuration system that allows fine-tuning for different document types, scanning conditions, and quality requirements.

## Configuration File Structure

The `config.toml` file is organized into logical sections:

```toml
[debug]     # Debugging and logging options
[processing]  # Core image processing parameters  
[cropping]   # Pre-processing and final cropping settings
[output]     # Output format and quality settings
```

## Section-by-Section Guide

### [debug] - Debugging and Logging

Controls intermediate file saving and logging behavior.

```toml
[debug]
save_intermediate = true
log_target = "stdout"
```

#### `save_intermediate` (boolean)
- **Default:** `true`
- **Purpose:** Save intermediate processing images for debugging
- **Files created:** `a_precrop.png`, `b_rotated.png`, `c_deskewed.png`, `d_stitched.png`
- **When to disable:** Production runs where disk space is limited
- **When to enable:** Debugging processing issues, understanding pipeline

#### `log_target` (string: "stdout" | "file")  
- **Default:** `"stdout"`
- **Purpose:** Control where detailed logs are written
- **"stdout":** Full detailed output to console (good for debugging)
- **"file":** Minimal console + detailed `processing.log` file (good for production)

**Example - Production setup:**
```toml
[debug]
save_intermediate = false
log_target = "file"
```

### [processing] - Core Processing Parameters

Controls the main image processing algorithms.

```toml
[processing]
angle_threshold = 0.1
max_rotation = 10.0
max_y_overlap = 1400
max_x_offset = 150
overlap_confidence_threshold = 0.25
overlap_confidence_backup = 0.20
```

#### `angle_threshold` (float, degrees)
- **Default:** `0.1`
- **Purpose:** Minimum rotation angle to trigger deskewing
- **Range:** `0.01` - `1.0`
- **Lower values:** More sensitive, corrects tiny rotations
- **Higher values:** Less sensitive, ignores minor skew
- **Tune when:** Documents appear slightly rotated but aren't being corrected

**Examples:**
```toml
angle_threshold = 0.05  # Very sensitive (old documents)
angle_threshold = 0.5   # Less sensitive (well-aligned scans)
```

#### `max_rotation` (float, degrees)
- **Default:** `10.0`
- **Purpose:** Maximum rotation angle to attempt correction
- **Range:** `1.0` - `45.0`
- **Purpose:** Prevents false positive corrections on severely rotated images
- **Tune when:** Getting "angle seems too large" messages

**Examples:**
```toml
max_rotation = 5.0   # Conservative (high-quality scans)
max_rotation = 15.0  # Aggressive (poor scanning conditions)
```

#### `max_y_overlap` (integer, pixels)
- **Default:** `1400`
- **Purpose:** Maximum vertical overlap to search when stitching pages
- **Range:** `500` - `2000`
- **Lower values:** Faster processing, works for small overlaps
- **Higher values:** Slower processing, handles large overlaps
- **Tune when:** Pages have very large or very small overlapping regions

#### `max_x_offset` (integer, pixels)
- **Default:** `150`
- **Purpose:** Maximum horizontal misalignment to correct when stitching
- **Range:** `50` - `300`
- **Accounts for:** Scanner feed variations, page positioning differences

#### `overlap_confidence_threshold` (float, 0.0-1.0)
- **Default:** `0.25`
- **Purpose:** Minimum matching confidence to accept overlap detection
- **Range:** `0.1` - `0.5`
- **Lower values:** More permissive, accepts weaker matches
- **Higher values:** More strict, requires better matches
- **Tune when:** Getting "No overlap found" errors

**Examples by document type:**
```toml
# High-quality typed documents
overlap_confidence_threshold = 0.35

# Handwritten documents  
overlap_confidence_threshold = 0.25

# Poor quality/faded documents
overlap_confidence_threshold = 0.15
```

#### `overlap_confidence_backup` (float, 0.0-1.0)
- **Default:** `0.20`
- **Purpose:** Fallback threshold when primary fails but position is consistent
- **Should be:** Lower than `overlap_confidence_threshold`

### [cropping] - Image Cropping Parameters

Controls pre-processing and final edge artifact removal.

```toml
[cropping]
precrop_max_vertical_percent = 10
```

#### `precrop_max_vertical_percent` (0-30)
- **Default:** `10`
- **Purpose:** Maximum percentage of image height to crop during pre-processing
- **Range:** `5` - `30`
- **Prevents:** Over-aggressive removal of manual rotation artifacts
- **Tune when:** Pre-processing is removing too much content

### [output] - Output Format Settings

Controls final image format and quality.

```toml
[output]
file_format = "png"
quality = 95       # note - this is only used by jpg!
```

#### `file_format` (string)
- **Options:** `"png"`, `"jpg"`, `"tiff"`
- **Default:** `"png"`

**Format comparison:**
- **PNG:** Lossless, good file size, recommended for most uses
- **JPG:** Lossy compression, smallest files, use for web/sharing
- **TIFF:** Lossless, largest files, use for archival storage

#### `quality` (integer, 1-100)
- **Default:** `95`
- **Purpose:** JPG compression quality (ignored for PNG/TIFF)
- **Range:** `85` - `100` for document preservation
- **Lower values:** Smaller files, visible compression artifacts
- **Higher values:** Larger files, better quality

## Configuration Templates

### High-Quality Modern Documents

```toml
[debug]
save_intermediate = false
log_target = "file"

[processing]
angle_threshold = 0.1
max_rotation = 5.0
overlap_confidence_threshold = 0.35

[cropping]
precrop_max_vertical_percent = 10

[output]
file_format = "png"
```

### Historical/Fragile Documents

```toml
[debug]
save_intermediate = true
log_target = "stdout"

[processing]
angle_threshold = 0.05
max_rotation = 8.0
overlap_confidence_threshold = 0.20

[cropping]
precrop_max_vertical_percent = 10

[output]
file_format = "png"
```

### Poor Quality Scans

```toml
[debug]
save_intermediate = true
log_target = "stdout"

[processing]
angle_threshold = 0.2
max_rotation = 15.0
overlap_confidence_threshold = 0.15
overlap_confidence_backup = 0.10

[cropping]
precrop_max_vertical_percent = 10

[output]
file_format = "png"
quality = 95
```

### Production/Batch Processing

```toml
[debug]
save_intermediate = false
log_target = "file"

[processing]
angle_threshold = 0.1
max_rotation = 10.0

[cropping]
precrop_max_vertical_percent = 10

[output]
file_format = "jpg"
quality = 92
```

## Troubleshooting by Symptom

### Text is being cut off
1. Check `precrop_max_vertical_percent` isn't too high

### No overlap detected
1. Decrease `overlap_confidence_threshold` to `0.15-0.20`
2. Decrease `overlap_confidence_backup` to `0.10`
3. Verify pages actually have overlapping content

### False rotation detection
1. Increase `angle_threshold` to `0.2-0.5`
2. Decrease `max_rotation` to `5.0`

### Rotation not being corrected
1. Decrease `angle_threshold` to `0.05`
2. Increase `max_rotation` to `15.0`

## Advanced Configuration

### Custom Scanning Setups

For specialized scanning equipment or unusual document conditions, you may need to experiment with parameters outside the recommended ranges. Always test with a small sample first.

### Debugging Workflow

1. Start with `save_intermediate = true` and `log_target = "stdout"`
2. Process a single pair to understand what's happening
3. Examine intermediate files to identify issues
4. Adjust relevant parameters
5. Disable debugging for production runs

### Performance Considerations

- `save_intermediate = false` saves disk space and processing time
- `log_target = "file"` reduces console noise in batch processing
- Lower `max_y_overlap` and `max_x_offset` speed up overlap detection
- JPG output significantly reduces file sizes for large batches
