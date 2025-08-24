import click
from pathlib import Path
import cv2
import numpy as np
import sys
import time

# Import shared logging utility
from logging_utils import setup_script_logging, get_script_logger

# Try to import tomllib (Python 3.11+) or fallback to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: Please install tomli package: pip install tomli")
        sys.exit(1)

# Script name for logging
SCRIPT_NAME = "stitch"

def load_config(config_path=None):
    """Load configuration from TOML file."""
    if config_path is None:
        config_path = Path("config.toml")
    
    try:
        with open(config_path, 'rb') as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Copy config.sample.toml to config.toml and edit as appropriate")
        sys.exit(1)

@click.command()
@click.argument('input_dir', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--config', type=click.Path(path_type=Path), help='Path to config.toml file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output for debugging')
def main(input_dir, output_dir, config, verbose):
    """Stitch paired document images together."""
    
    # Early messages before logging is set up
    print(f"Starting image stitching script...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load configuration
    try:
        config_data = load_config(config)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return
    
    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory created: {output_dir}")
    except Exception as e:
        print(f"✗ Failed to create output directory: {e}")
        return
    
    # Setup logging using shared utility
    logger = setup_script_logging(SCRIPT_NAME, config_data, output_dir, verbose)
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log target: {config_data['debug']['log_target']}")
    logger.info(f"Stitching config: max_y_overlap={config_data['stitching']['max_y_overlap']}px, max_x_offset={config_data['stitching']['max_x_offset']}px")
    logger.info(f"Confidence thresholds: primary={config_data['stitching']['overlap_confidence_threshold']}, backup={config_data['stitching']['overlap_confidence_backup']}")
    
    # Find image files
    logger.info("Scanning for image files...")
    image_files = []
    
    try:
        for file in input_dir.iterdir():
            if file.suffix.lower() == '.png':
                image_files.append(file)
                logger.debug(f"  Found: {file.name}")
    except Exception as e:
        logger.error(f"Error scanning input directory: {e}")
        return
    
    image_files.sort()  # Alphabetical ordering
    
    if not image_files:
        logger.error("No PNG image files found in input directory")
        logger.error(f"Looking for files with extension: .png")
        return
    
    # Pair them up
    pairs = []
    for i in range(0, len(image_files), 2):
        if i + 1 < len(image_files):
            pairs.append((image_files[i], image_files[i + 1]))
        else:
            logger.warning(f"Odd number of files - skipping last file: {image_files[i].name}")
    
    if not pairs:
        logger.error("Need at least 2 images to create pairs")
        return
    
    logger.info(f"Found {len(image_files)} image files, created {len(pairs)} pairs")
    for i, (file1, file2) in enumerate(pairs, 1):
        logger.debug(f"  Pair {i}: {file1.name} + {file2.name}")
    
    # Process each pair
    processed_count = 0
    failed_count = 0
    
    for i, (file1, file2) in enumerate(pairs, 1):
        pair_msg = f"Processing pair {i}/{len(pairs)}: {file1.name} + {file2.name}"
        logger.info(pair_msg)
        logger.debug(f"\n{pair_msg}")
        
        start_time = time.time()
        
        try:
            # Load images with OpenCV
            logger.debug(f"  Loading images...")
            img1 = cv2.imread(str(file1))  # Top page
            img2 = cv2.imread(str(file2))  # Bottom page
            
            if img1 is None or img2 is None:
                logger.error(f"  Could not load one or both images")
                failed_count += 1
                continue
            
            logger.debug(f"  ✓ Loaded images: {img1.shape[1]}x{img1.shape[0]} (top page) and {img2.shape[1]}x{img2.shape[0]} (bottom page)")
            
            # Stitch the images - put top page (img1) on top, bottom page (img2) on bottom
            logger.debug(f"  Stitching images (top page on top, bottom page on bottom):")
            stitched_img, overlap_info = stitch_images(img1, img2, config_data)
            
            # Save final stitched result
            logger.debug(f"  Saving stitched image...")
            page_num_str = f"{i:03d}"
            final_path = output_dir / f"page{page_num_str}.png"
            cv2.imwrite(str(final_path), stitched_img)
            
            elapsed_time = time.time() - start_time
            final_msg = f"✓ Completed in {elapsed_time:.2f}s - {overlap_info} - saved as {final_path.name}"
            logger.info(final_msg)
            logger.debug(f"  {final_msg}")
            processed_count += 1
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"✗ Error processing pair {i} after {elapsed_time:.2f}s: {e}"
            logger.error(error_msg)
            logger.debug(f"  {error_msg}")
            import traceback
            logger.debug(traceback.format_exc())
            failed_count += 1
            continue
    
    # Summary
    logger.info(f"\n✨ Stitching complete!")
    logger.info(f"Successfully processed: {processed_count} pairs")
    if failed_count > 0:
        logger.error(f"Failed to process: {failed_count} pairs")
    logger.info(f"Results saved to: {output_dir}")

def stitch_images(top_img, bottom_img, config):
    """
    Stitch two images together, either with overlap detection or simple concatenation.
    Returns tuple of (stitched_image, overlap_info_string)
    """
    logger = get_script_logger(SCRIPT_NAME)
    
    h1, w1 = top_img.shape[:2]
    h2, w2 = bottom_img.shape[:2]
    
    logger.debug(f"    Image 1 (top): {w1}x{h1}")
    logger.debug(f"    Image 2 (bottom): {w2}x{h2}")
    
    # Find the best overlap using template matching
    overlap_result = find_overlap_between_images(top_img, bottom_img, config)
    
    if overlap_result is None:
        logger.debug(f"    No good overlap found, using simple concatenation")
        stitched = concatenate_images(top_img, bottom_img)
        return stitched, "concatenated (no overlap detected)"
    
    y_overlap, x_offset, confidence = overlap_result
    logger.debug(f"    Overlap found: Y={y_overlap}px, X={x_offset}px (confidence: {confidence:.3f})")
    
    # Create stitched image directly without any scaling
    stitched = create_stitched_image(top_img, bottom_img, y_overlap, x_offset)
    overlap_info = f"overlap Y={y_overlap}px, X={x_offset}px, conf={confidence:.3f}"
    return stitched, overlap_info

def find_overlap_between_images(top_img, bottom_img, config):
    """
    Find overlap using simple template matching with a small sample from bottom image.
    """
    logger = get_script_logger(SCRIPT_NAME)
    
    # Get processing config
    max_y_overlap = config['stitching']['max_y_overlap']
    max_x_offset = config['stitching']['max_x_offset']
    confidence_threshold = config['stitching']['overlap_confidence_threshold']
    backup_threshold = config['stitching']['overlap_confidence_backup']
    
    # Convert to grayscale for matching
    top_gray = cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY) if len(top_img.shape) == 3 else top_img
    bottom_gray = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2GRAY) if len(bottom_img.shape) == 3 else bottom_img
    
    # Template parameters - take a sample from top of bottom image
    template_height = 200
    template_width = 400
    
    # Extract template from top-center of bottom image
    bottom_h, bottom_w = bottom_gray.shape
    template_x_start = max(0, (bottom_w - template_width) // 2)
    template_x_end = min(bottom_w, template_x_start + template_width)
    template = bottom_gray[0:template_height, template_x_start:template_x_end]
    
    logger.debug(f"    Template: {template.shape[1]}x{template.shape[0]} from bottom image at x={template_x_start}-{template_x_end}")
    
    # Search area in top image - bottom portion only (use config value)
    search_height = min(max_y_overlap, top_img.shape[0])  # Use config parameter
    search_area = top_gray[-search_height:, :]
    
    logger.debug(f"    Search area: {search_area.shape[1]}x{search_area.shape[0]} (bottom {search_height}px of top image)")
    
    # Perform template matching
    try:
        result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        logger.debug(f"    Template matching: max_val={max_val:.3f} at location {max_loc}")
        
        # Convert match location to actual overlap parameters
        match_x, match_y = max_loc
        
        # Calculate Y overlap - how far up from bottom of top image was the match found
        y_overlap = search_height - match_y
        
        # Calculate X offset - difference between template position and match position
        template_center_x = template_x_start + template.shape[1] // 2
        match_center_x = match_x + template.shape[1] // 2
        x_offset = match_center_x - template_center_x
        
        logger.debug(f"    Calculated overlap: Y={y_overlap}px, X={x_offset}px (confidence: {max_val:.3f})")
        logger.debug(f"    Template center: {template_center_x}, Match center: {match_center_x}, Offset: {x_offset}")
        
        # Check if X offset is within acceptable range
        if abs(x_offset) > max_x_offset:
            logger.debug(f"    ✗ X offset {x_offset}px exceeds maximum {max_x_offset}px - rejecting match")
            return None
        
        # Check if confidence is good enough
        if max_val > confidence_threshold:
            logger.debug(f"    ✓ Good match found: Y={y_overlap}px, X={x_offset}px (confidence: {max_val:.3f})")
            return y_overlap, x_offset, max_val
        elif max_val > backup_threshold:
            logger.warning(f"    ⚠ Acceptable match found: Y={y_overlap}px, X={x_offset}px (confidence: {max_val:.3f})")
            return y_overlap, x_offset, max_val
        else:
            logger.debug(f"    ✗ Match confidence too low: {max_val:.3f} < {confidence_threshold}")
            return None
            
    except Exception as e:
        logger.error(f"    Template matching failed: {e}")
        return None

def create_stitched_image(top_img, bottom_img, y_overlap, x_offset):
    """
    Create the final stitched image with hard cut (no blending).
    """
    logger = get_script_logger(SCRIPT_NAME)
    
    top_h, top_w = top_img.shape[:2]
    bottom_h, bottom_w = bottom_img.shape[:2]
    
    # Calculate canvas dimensions
    if x_offset >= 0:
        canvas_width = max(top_w, bottom_w + x_offset)
        bottom_x_start = x_offset
        top_x_start = 0
    else:
        canvas_width = max(top_w - x_offset, bottom_w)
        bottom_x_start = 0
        top_x_start = -x_offset
    
    canvas_height = top_h + bottom_h - y_overlap
    
    # Create white canvas
    stitched = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    logger.debug(f"    Canvas size: {canvas_width}x{canvas_height}")
    logger.debug(f"    Top image position: ({top_x_start}, 0)")
    logger.debug(f"    Bottom image position: ({bottom_x_start}, {top_h - y_overlap})")
    
    # Place top image completely first
    stitched[:top_h, top_x_start:top_x_start + top_w] = top_img
    
    # Place bottom image, overwriting the overlap region
    bottom_y_start = top_h - y_overlap
    stitched[bottom_y_start:bottom_y_start + bottom_h, bottom_x_start:bottom_x_start + bottom_w] = bottom_img
    
    logger.debug(f"    ✓ Stitching complete: {stitched.shape[1]}x{stitched.shape[0]}")
    
    return stitched

def concatenate_images(top_img, bottom_img):
    """Simple concatenation fallback when overlap detection fails."""
    logger = get_script_logger(SCRIPT_NAME)
    
    # Make images the same width
    target_width = max(top_img.shape[1], bottom_img.shape[1])
    
    if top_img.shape[1] != target_width:
        top_resized = cv2.resize(top_img, (target_width, top_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        top_resized = top_img
        
    if bottom_img.shape[1] != target_width:
        bottom_resized = cv2.resize(bottom_img, (target_width, bottom_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        bottom_resized = bottom_img
    
    stitched = np.vstack([top_resized, bottom_resized])
    
    logger.debug(f"    ✓ Simple concatenation: {stitched.shape[1]}x{stitched.shape[0]}")
    return stitched

if __name__ == '__main__':
    main()
