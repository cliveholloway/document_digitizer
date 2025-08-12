import click
from rich.console import Console
from pathlib import Path
import shutil
from PIL import Image
import cv2
import numpy as np
import logging
import sys

# Try to import tomllib (Python 3.11+) or fallback to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: Please install tomli package: pip install tomli")
        sys.exit(1)

def load_config(config_path=None):
    """Load configuration from TOML file."""
    if config_path is None:
        config_path = Path("config.toml")
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please create a config.toml file or specify path with --config")
        sys.exit(1)
    
    try:
        with open(config_path, 'rb') as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def setup_logging(config, output_dir):
    """Setup logging based on configuration."""
    log_target = config['debug']['log_target']
    
    if log_target == 'file':
        # Setup file logging with minimal console output
        log_file = output_dir / 'processing.log'
        
        # Configure file logger
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[file_handler]
        )
        
        # Create console logger for minimal output
        console_logger = logging.getLogger('console')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        console_logger.addHandler(console_handler)
        console_logger.setLevel(logging.INFO)
        console_logger.propagate = False
        
        print(f"Detailed logging to: {log_file}")
        return logging.getLogger(), console_logger
    else:
        # Stdout logging - return Rich console wrapped as logger
        return None, None

@click.command()
@click.argument('input_dir', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--config', type=click.Path(path_type=Path), help='Path to config.toml file')
def process(input_dir, output_dir, config):
    """Process paired handwriting images into transcribed text."""
    
    # Load configuration
    config_data = load_config(config)
    
    console = Console()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    file_logger, console_logger = setup_logging(config_data, output_dir)
    
    # Choose logging method based on config
    if config_data['debug']['log_target'] == 'file':
        def log_info(msg):
            file_logger.info(msg)
        def log_debug(msg):
            file_logger.debug(msg)
        def console_info(msg):
            console_logger.info(msg)
    else:
        def log_info(msg):
            console.print(msg)
        def log_debug(msg):
            console.print(msg)
        def console_info(msg):
            console.print(msg)
    
    log_info(f"[green]Input directory:[/green] {input_dir}")
    log_info(f"[green]Output directory:[/green] {output_dir}")
    log_info(f"[green]Debug mode:[/green] {'enabled' if config_data['debug']['save_intermediate'] else 'disabled'}")
    log_info(f"[green]Log target:[/green] {config_data['debug']['log_target']}")
    log_info(f"[green]Processing config:[/green] angle_threshold={config_data['processing']['angle_threshold']}°, max_rotation={config_data['processing']['max_rotation']}°")
    log_info(f"[green]Output config:[/green] format={config_data['output']['file_format']}, quality={config_data['output']['quality']}")
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = []
    
    for file in input_dir.iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    image_files.sort()  # Alphabetical ordering
    
    log_info(f"[blue]Found {len(image_files)} image files:[/blue]")
    for file in image_files:
        log_debug(f"  • {file.name}")
    
    # Pair them up
    pairs = []
    for i in range(0, len(image_files), 2):
        if i + 1 < len(image_files):
            pairs.append((image_files[i], image_files[i + 1]))
    
    log_info(f"[yellow]Created {len(pairs)} pairs[/yellow]")
    for i, (file1, file2) in enumerate(pairs, 1):
        log_debug(f"  Pair {i}: {file1.name} + {file2.name}")
    
    # Process each pair
    with console.status("[bold green]Processing image pairs...") if config_data['debug']['log_target'] == 'stdout' else console.status(""):
        for i, (file1, file2) in enumerate(pairs, 1):
            pair_msg = f"Processing pair {i}: {file1.name} + {file2.name}"
            console_info(pair_msg)
            log_debug(f"\n[cyan]{pair_msg}[/cyan]")
            
            try:
                # Load images with OpenCV
                img1_cv = cv2.imread(str(file1))
                img2_cv = cv2.imread(str(file2))
                
                if img1_cv is None or img2_cv is None:
                    log_debug(f"  [red]Error: Could not load one or both images[/red]")
                    continue
                
                log_debug(f"  ✓ Loaded images: {img1_cv.shape[1]}x{img1_cv.shape[0]} and {img2_cv.shape[1]}x{img2_cv.shape[0]}")
                
                # Pre-process: crop to content area (remove manual rotation artifacts)
                log_debug(f"  Pre-processing image 1 (crop to content):")
                img1_precropped = crop_to_content_area(img1_cv, log_debug, "image 1", config_data)
                
                log_debug(f"  Pre-processing image 2 (crop to content):")
                img2_precropped = crop_to_content_area(img2_cv, log_debug, "image 2", config_data)
                
                # Save debug images: a_precrop
                if config_data['debug']['save_intermediate']:
                    cv2.imwrite(str(output_dir / f"pair_{i:03d}_a_page1_precrop.png"), img1_precropped)
                    cv2.imwrite(str(output_dir / f"pair_{i:03d}_a_page2_precrop.png"), img2_precropped)
                
                # Deskew both images (now includes automatic mathematical cropping)
                log_debug(f"  Deskewing and cropping image 1:")
                img1_processed, img1_rotated = deskew_image_improved(img1_precropped, log_debug, config_data['processing'])
                
                log_debug(f"  Deskewing and cropping image 2:")
                img2_processed, img2_rotated = deskew_image_improved(img2_precropped, log_debug, config_data['processing'])
                
                # Save debug images: b_rotated and c_deskewed
                if config_data['debug']['save_intermediate']:
                    cv2.imwrite(str(output_dir / f"pair_{i:03d}_b_page1_rotated.png"), img1_rotated)
                    cv2.imwrite(str(output_dir / f"pair_{i:03d}_b_page2_rotated.png"), img2_rotated)
                    cv2.imwrite(str(output_dir / f"pair_{i:03d}_c_page1_deskewed.png"), img1_processed)
                    cv2.imwrite(str(output_dir / f"pair_{i:03d}_c_page2_deskewed.png"), img2_processed)
                
                # Stitch the processed images with NO BLENDING
                log_debug(f"  Stitching images (NO BLENDING):")
                stitched_img, x_offset_used = stitch_images_no_blend(img1_processed, img2_processed, log_debug, config_data)
                
                # Save debug image: d_stitched
                if config_data['debug']['save_intermediate']:
                    cv2.imwrite(str(output_dir / f"pair_{i:03d}_d_stitched.png"), stitched_img)
                
                # Final crop to remove edge artifacts based on stitching offset
                log_debug(f"  Final edge cleanup crop:")
                final_img = final_crop_to_document(stitched_img, x_offset_used, log_debug)
                
                # Save final result with configured format
                output_format = config_data['output']['file_format'].lower()
                if output_format == 'jpg' or output_format == 'jpeg':
                    final_path = output_dir / f"pair_{i:03d}.jpg"
                    cv2.imwrite(str(final_path), final_img, [cv2.IMWRITE_JPEG_QUALITY, config_data['output']['quality']])
                elif output_format == 'tiff' or output_format == 'tif':
                    final_path = output_dir / f"pair_{i:03d}.tiff"
                    cv2.imwrite(str(final_path), final_img)
                else:  # Default to PNG
                    final_path = output_dir / f"pair_{i:03d}.png"
                    cv2.imwrite(str(final_path), final_img)
                
                final_msg = f"Processed image saved as {final_path.name}"
                console_info(final_msg)
                log_debug(f"  ✓ {final_msg}")
                
            except Exception as e:
                error_msg = f"Error processing pair {i}: {e}"
                console_info(f"[red]{error_msg}[/red]")
                log_debug(f"  [red]{error_msg}[/red]")
                import traceback
                log_debug(traceback.format_exc())
                continue
    
    log_info(f"\n[bold green]✨ Processing complete![/bold green]")
    log_info(f"Results saved to: {output_dir}")

def crop_to_content_area(img_array, log_func, img_name, config):
    """
    Crop using systematic edge detection:
    1. Crop inward from left/right while columns are only background
    2. Crop from top/bottom based on edge column analysis
    """
    log_func(f"    Analyzing {img_name} with systematic edge cropping...")
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    h, w = gray.shape
    log_func(f"    Original size: {w}x{h}")
    
    # Get cropping config
    max_vertical_percent = config['cropping']['precrop_max_vertical_percent']
    
    # Define what counts as "background" (transparent, black, or white)
    def is_background_pixel(pixel_value):
        return pixel_value < 30 or pixel_value > 240  # Black or white
    
    def is_background_column(col_pixels):
        return np.all([is_background_pixel(p) for p in col_pixels])
    
    # Step 1: Crop from left while columns are purely background
    left_crop = 0
    for x in range(w):
        column = gray[:, x]
        if is_background_column(column):
            left_crop = x + 1
            log_func(f"    Column {x}: all background, cropping")
        else:
            log_func(f"    Column {x}: has content, stopping left crop")
            break
    
    # Step 1: Crop from right while columns are purely background  
    right_crop = w
    for x in range(w-1, -1, -1):
        column = gray[:, x]
        if is_background_column(column):
            right_crop = x
            log_func(f"    Column {x}: all background, cropping")
        else:
            log_func(f"    Column {x}: has content, stopping right crop")
            break
    
    log_func(f"    Horizontal crop: left={left_crop}, right={right_crop} (width: {w} → {right_crop - left_crop})")
    
    # Apply horizontal crop first
    if left_crop < right_crop:
        gray_h_cropped = gray[:, left_crop:right_crop]
        img_h_cropped = img_array[:, left_crop:right_crop]
    else:
        log_func(f"    ⚠ No valid horizontal crop, keeping original")
        return img_array
    
    # Now work on the horizontally cropped image
    h_new, w_new = gray_h_cropped.shape
    log_func(f"    After horizontal crop: {w_new}x{h_new}")
    
    # Step 2: Analyze edge columns for vertical cropping
    max_vertical_crop = int(h_new * max_vertical_percent/100)
    log_func(f"    Max vertical crop allowed: {max_vertical_crop}px ({max_vertical_percent:.1f}%)")
    
    # Check first column (leftmost after horizontal crop)
    first_col = gray_h_cropped[:, 0]
    last_col = gray_h_cropped[:, -1]
    
    # Step 2a: Crop from top based on edge column analysis
    top_crop = 0
    for y in range(min(max_vertical_crop, h_new)):
        first_col_pixel = first_col[y]
        last_col_pixel = last_col[y]
        
        # If BOTH edge pixels are background, we can crop this row
        if is_background_pixel(first_col_pixel) and is_background_pixel(last_col_pixel):
            top_crop = y + 1
            if y % 20 == 0:  # Debug every 20 rows
                log_func(f"    Row {y}: edges are background ({first_col_pixel:.0f}, {last_col_pixel:.0f})")
        else:
            log_func(f"    Row {y}: content detected, stopping top crop ({first_col_pixel:.0f}, {last_col_pixel:.0f})")
            break
    
    # Step 2b: Crop from bottom based on edge column analysis  
    bottom_crop = h_new
    for y in range(h_new-1, max(h_new - max_vertical_crop - 1, -1), -1):
        first_col_pixel = first_col[y]
        last_col_pixel = last_col[y]
        
        # If BOTH edge pixels are background, we can crop this row
        if is_background_pixel(first_col_pixel) and is_background_pixel(last_col_pixel):
            bottom_crop = y
            if (h_new - y) % 20 == 0:  # Debug every 20 rows
                log_func(f"    Row {y}: edges are background ({first_col_pixel:.0f}, {last_col_pixel:.0f})")
        else:
            log_func(f"    Row {y}: content detected, stopping bottom crop ({first_col_pixel:.0f}, {last_col_pixel:.0f})")
            break
    
    log_func(f"    Vertical crop: top={top_crop}, bottom={bottom_crop} (height: {h_new} → {bottom_crop - top_crop})")
    
    # Apply vertical crop
    if top_crop < bottom_crop:
        final_cropped = img_h_cropped[top_crop:bottom_crop, :]
    else:
        log_func(f"    ⚠ No valid vertical crop, using horizontal crop only")
        final_cropped = img_h_cropped
    
    # Calculate final coordinates for reporting
    final_h, final_w = final_cropped.shape[:2]
    total_left = left_crop
    total_right = left_crop + final_w
    total_top = top_crop  
    total_bottom = top_crop + final_h
    
    log_func(f"    ✓ Systematic crop complete:")
    log_func(f"      Original: {w}x{h}")
    log_func(f"      Final: {final_w}x{final_h}")
    log_func(f"      Removed: {w-final_w}px width, {h-final_h}px height")
    log_func(f"      Boundaries: X={total_left}-{total_right}, Y={total_top}-{total_bottom}")
    
    return final_cropped

def deskew_image_improved(img_array, log_func, processing_config):
    """
    Conservative deskewing that handles a reasonable range of rotations.
    Now includes automatic cropping based on rotation angle.
    """
    angle_threshold = processing_config['angle_threshold']
    max_rotation = processing_config['max_rotation']
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()
    
    # Use a single, reliable method: Hough Line Transform with conservative settings
    detected_angle = detect_angle_conservative(gray, log_func)
    
    if detected_angle is None:
        log_func(f"    No significant rotation detected")
        return img_array, img_array
    
    log_func(f"    Detected angle: {detected_angle:.3f}°")
    
    if abs(detected_angle) < angle_threshold:
        log_func(f"    Angle {detected_angle:.3f}° < threshold {angle_threshold}°. Skipping rotation.")
        return img_array, img_array
    
    if abs(detected_angle) > max_rotation:
        log_func(f"    Angle {detected_angle:.3f}° > max rotation {max_rotation}°, skipping to avoid overcorrection.")
        return img_array, img_array
    
    # Rotate the image with proper boundary handling
    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate rotation matrix
    M = cv2.getRotationMatrix2D(center, detected_angle, 1.0)
    
    # Calculate new image dimensions to prevent cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation with white background
    rotated = cv2.warpAffine(img_array, M, (new_w, new_h), 
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
    
    log_func(f"    ✓ Rotated by {detected_angle:.3f}° (size: {w}x{h} → {new_w}x{new_h})")
    
    # Now crop based on the rotation angle to remove the white wedges
    cropped = crop_rotation_mathematically(rotated, detected_angle, w, h, log_func)
    
    return cropped, rotated  # Return both for debugging

def detect_angle_conservative(gray, log_func):
    """
    Conservative angle detection with multiple fallback methods and detailed debugging.
    """
    # Apply mild blur to reduce noise but preserve line structure
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use conservative edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    log_func(f"    Image shape: {gray.shape}, Edge pixels: {np.sum(edges > 0)}")
    
    # Try multiple detection methods with increasing aggressiveness
    methods_tried = []
    
    # Method 1: Conservative probabilistic
    log_func(f"    Method 1: Conservative probabilistic Hough...")
    angle1 = try_probabilistic_hough_conservative(edges, log_func)
    if angle1 is not None:
        methods_tried.append(('Conservative Prob', angle1))
    
    # Method 2: Aggressive probabilistic
    log_func(f"    Method 2: Aggressive probabilistic Hough...")
    angle2 = try_probabilistic_hough_aggressive(edges, log_func)
    if angle2 is not None:
        methods_tried.append(('Aggressive Prob', angle2))
    
    # Method 3: Standard Hough
    log_func(f"    Method 3: Standard Hough...")
    angle3 = try_standard_hough_flexible(edges, log_func)
    if angle3 is not None:
        methods_tried.append(('Standard', angle3))
    
    # Method 4: Projection method (last resort)
    log_func(f"    Method 4: Projection profile...")
    angle4 = try_projection_method(gray, log_func)
    if angle4 is not None:
        methods_tried.append(('Projection', angle4))
    
    if not methods_tried:
        log_func(f"    [red]ERROR: No method detected any angles![/red]")
        return None
    
    # Display all results
    log_func(f"    Results from {len(methods_tried)} methods:")
    for method, angle in methods_tried:
        log_func(f"      {method}: {angle:.3f}°")
    
    # Choose the best result
    if len(methods_tried) == 1:
        final_angle = methods_tried[0][1]
        log_func(f"    Using only available result: {final_angle:.3f}°")
    else:
        # Filter out any obviously wrong results (> 15°)
        reasonable_angles = [(m, a) for m, a in methods_tried if abs(a) <= 15]
        
        if not reasonable_angles:
            log_func(f"    All detected angles > 15°, skipping")
            return None
        
        # If multiple reasonable results, take median
        angles = [a for _, a in reasonable_angles]
        final_angle = np.median(angles)
        log_func(f"    Using median of {len(reasonable_angles)} reasonable results: {final_angle:.3f}°")
    
    return final_angle

def try_probabilistic_hough_conservative(edges, log_func):
    """Conservative probabilistic Hough - same as before."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=150,
        maxLineGap=30
    )
    
    return process_probabilistic_lines(lines, log_func, "conservative")

def try_probabilistic_hough_aggressive(edges, log_func):
    """More aggressive probabilistic Hough to catch subtle lines."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,  # Higher angular resolution
        threshold=15,       # Much lower threshold
        minLineLength=80,   # Shorter lines
        maxLineGap=50      # Larger gaps
    )
    
    return process_probabilistic_lines(lines, log_func, "aggressive")

def process_probabilistic_lines(lines, log_func, method_name):
    """Process probabilistic Hough lines with less aggressive outlier removal."""
    if lines is None:
        log_func(f"      No lines found with {method_name} probabilistic method")
        return None
    
    angles = []
    line_details = []
    
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) < 5:
            continue
            
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize angle
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        
        # Accept wider range for debugging
        if abs(angle) <= 20:
            angles.append(angle)
            length = np.sqrt(dx*dx + dy*dy)
            line_details.append(f"({x1},{y1})-({x2},{y2}) len={length:.0f} ang={angle:.1f}°")
    
    log_func(f"      Found {len(angles)} lines with {method_name} method")
    
    if len(angles) >= 2:  # Lowered requirement
        # Show some line details for debugging
        for i, detail in enumerate(line_details[:5]):  # Show more lines
            log_func(f"        Line {i+1}: {detail}")
        if len(line_details) > 5:
            log_func(f"        ... and {len(line_details)-5} more")
        
        # Calculate angle statistics before filtering
        mean_angle = np.mean(angles)
        median_angle = np.median(angles)
        std_angle = np.std(angles)
        
        log_func(f"      Raw stats: mean={mean_angle:.3f}°, median={median_angle:.3f}°, std={std_angle:.3f}°")
        
        # Less aggressive outlier removal - only if we have many lines
        if len(angles) > 10:
            filtered_angles = remove_outliers_gentle(angles)
        else:
            filtered_angles = angles
            
        final_median = np.median(filtered_angles)
        log_func(f"      {method_name.capitalize()} result: {final_median:.3f}° from {len(filtered_angles)} lines (filtered from {len(angles)})")
        
        # If the median is very close to 0 but we see -1° lines, investigate further
        if abs(final_median) < 0.1 and any(abs(a) > 0.5 for a in angles):
            # Count significant angles
            significant_angles = [a for a in angles if abs(a) > 0.3]
            if len(significant_angles) > len(angles) * 0.1:  # If >10% of lines show rotation
                significant_median = np.median(significant_angles)
                log_func(f"      Found {len(significant_angles)} significant angles, median: {significant_median:.3f}°")
                return significant_median
        
        return final_median
    
    log_func(f"      Not enough lines for {method_name} method")
    return None

def remove_outliers_gentle(angles):
    """More gentle outlier removal that preserves consistent small rotations."""
    if len(angles) < 6:
        return angles
    
    # Use a wider IQR multiplier to be less aggressive
    q1, q3 = np.percentile(angles, [20, 80])  # Use 20th and 80th percentiles instead of 25th/75th
    iqr = q3 - q1
    lower_bound = q1 - 2.0 * iqr  # Use 2.0 instead of 1.5
    upper_bound = q3 + 2.0 * iqr
    
    filtered = [a for a in angles if lower_bound <= a <= upper_bound]
    
    # Don't filter too aggressively - keep at least 50% of the data
    if len(filtered) < len(angles) * 0.5:
        return angles
    
    return filtered

def try_standard_hough_flexible(edges, log_func):
    """Standard Hough with more flexible parameters."""
    # Try with lower threshold first
    for threshold in [40, 25, 15]:
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=threshold)
        
        if lines is not None and len(lines) >= 3:
            log_func(f"      Standard Hough found {len(lines)} lines with threshold {threshold}")
            
            angles = []
            for rho, theta in lines[:, 0]:
                angle_degrees = np.degrees(theta) - 90
                if abs(angle_degrees) <= 20:  # Wider acceptance
                    angles.append(angle_degrees)
            
            if len(angles) >= 2:
                filtered_angles = remove_outliers(angles) if len(angles) > 3 else angles
                median_angle = np.median(filtered_angles)
                log_func(f"      Standard result: {median_angle:.3f}° from {len(filtered_angles)} lines")
                return median_angle
    
    log_func(f"      Standard Hough found no suitable lines")
    return None

def remove_outliers(angles):
    """Remove outliers using interquartile range method."""
    if len(angles) < 4:
        return angles
        
    q1, q3 = np.percentile(angles, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered = [a for a in angles if lower_bound <= a <= upper_bound]
    return filtered

def try_projection_method(gray, log_func):
    """Projection profile method as last resort."""
    h, w = gray.shape
    
    # Try smaller angle range with finer steps
    angles_to_try = np.arange(-8, 8.1, 0.2)
    best_angle = 0
    best_score = 0
    
    log_func(f"      Testing {len(angles_to_try)} angles...")
    
    for angle in angles_to_try:
        # Rotate image
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), borderValue=255)
        
        # Calculate horizontal projection
        projection = np.sum(rotated < 200, axis=1)
        
        # Calculate variance (higher = better text line alignment)
        if len(projection) > 0:
            score = np.var(projection)
            if score > best_score:
                best_score = score
                best_angle = angle
    
    log_func(f"      Best projection angle: {best_angle:.3f}° (score: {best_score:.0f})")
    
    # Lower threshold for acceptance
    if best_score > 500:  # Much lower threshold
        return best_angle
    else:
        log_func(f"      Projection score too low ({best_score:.0f} < 500)")
        return None

def crop_rotation_mathematically(rotated_img, angle_degrees, original_w, original_h, log_func):
    """
    Crop the rotated image based on the exact rotation angle using correct trigonometry.
    Now with more conservative cropping to avoid cutting content.
    """
    if abs(angle_degrees) < 0.1:  # No significant rotation
        return rotated_img
    
    # Convert angle to radians
    angle_rad = np.radians(abs(angle_degrees))
    
    # Calculate theoretical crop amounts
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Theoretical full crop amounts
    crop_horizontal_full = int(original_h * sin_a)
    crop_vertical_full = int(original_w * sin_a)
    
    # Conservative on horizontal (preserve text width), full theoretical on vertical
    horizontal_safety_factor = 0.8  # Conservative on sides (preserve text width)
    vertical_safety_factor = 1.0    # Full theoretical amount for top/bottom
    
    crop_horizontal = int(crop_horizontal_full * horizontal_safety_factor)
    crop_vertical = int(crop_vertical_full * vertical_safety_factor)
    
    log_func(f"    Mathematical crop calculation (full vertical):")
    log_func(f"      Angle: {angle_degrees:.3f}° = {np.degrees(angle_rad):.3f}°")
    log_func(f"      Original: {original_w}x{original_h}")
    log_func(f"      sin({angle_degrees:.3f}°) = {sin_a:.4f}")
    log_func(f"      Theoretical crop: {crop_horizontal_full}px left/right, {crop_vertical_full}px top/bottom")
    log_func(f"      Applied crop: {crop_horizontal}px left/right ({horizontal_safety_factor}x), {crop_vertical}px top/bottom ({vertical_safety_factor}x)")
    
    # Get current dimensions
    curr_h, curr_w = rotated_img.shape[:2]
    
    # Calculate crop bounds
    left = crop_horizontal
    right = curr_w - crop_horizontal
    top = crop_vertical
    bottom = curr_h - crop_vertical
    
    log_func(f"      Current rotated size: {curr_w}x{curr_h}")
    log_func(f"      Crop bounds: left={left}, right={right}, top={top}, bottom={bottom}")
    log_func(f"      Final size will be: {right-left}x{bottom-top}")
    
    # Safety check - make sure we're not cropping more than the image
    if left >= right or top >= bottom:
        log_func(f"    ⚠ Calculated crop would remove entire image, skipping crop")
        return rotated_img
    
    # Additional safety check - don't crop more than 15% from any edge
    max_crop_h = int(curr_h * 0.15)
    max_crop_w = int(curr_w * 0.15)
    
    if crop_vertical > max_crop_h:
        log_func(f"    ⚠ Limiting vertical crop from {crop_vertical} to {max_crop_h}")
        top = max_crop_h
        bottom = curr_h - max_crop_h
    
    if crop_horizontal > max_crop_w:
        log_func(f"    ⚠ Limiting horizontal crop from {crop_horizontal} to {max_crop_w}")
        left = max_crop_w
        right = curr_w - max_crop_w
    
    # Apply the crop
    cropped = rotated_img[top:bottom, left:right]
    
    log_func(f"    ✓ Conservatively cropped: {curr_w}x{curr_h} → {cropped.shape[1]}x{cropped.shape[0]}")
    
    return cropped

def stitch_images_no_blend(img1_array, img2_array, log_func, config):
    """
    Stitch images with NO BLENDING - hard cut to preserve text quality.
    Returns both the stitched image and the x_offset used for final cropping.
    """
    h1, w1 = img1_array.shape[:2]
    h2, w2 = img2_array.shape[:2]
    
    log_func(f"    Image 1: {w1}x{h1}")
    log_func(f"    Image 2: {w2}x{h2}")
    
    # Always treat first image as top, second as bottom
    top_img = img1_array
    bottom_img = img2_array
    
    # Find the best overlap using template matching
    overlap_result = find_overlap_template_matching(top_img, bottom_img, log_func, config)
    
    if overlap_result is None:
        log_func(f"    No good overlap found, using simple concatenation")
        stitched = simple_concatenate(top_img, bottom_img, log_func)
        return stitched, 0  # Return 0 x_offset for simple concatenation
    
    y_overlap, x_offset, confidence = overlap_result
    log_func(f"    Best overlap: Y={y_overlap}px, X={x_offset}px (confidence: {confidence:.3f})")
    
    # Create stitched image with NO BLENDING
    stitched = create_stitched_image_no_blend(top_img, bottom_img, y_overlap, x_offset, log_func)
    return stitched, x_offset  # Return both image and x_offset

def find_overlap_template_matching(top_img, bottom_img, log_func, config):
    """
    Find overlap using template matching optimized for full-size document pages with content overlap.
    """
    # Get processing config
    max_y_overlap = config['processing']['max_y_overlap']
    max_x_offset = config['processing']['max_x_offset']
    confidence_threshold = config['processing']['overlap_confidence_threshold']
    backup_threshold = config['processing']['overlap_confidence_backup']
    
    # Convert to grayscale for matching
    top_gray = cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY) if len(top_img.shape) == 3 else top_img
    bottom_gray = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2GRAY) if len(bottom_img.shape) == 3 else bottom_img
    
    best_score = 0
    best_overlap = 0
    best_offset = 0
    
    # For full-size document pages - much larger search range
    y_start = max(100, min(200, bottom_img.shape[0] // 4))
    y_end = min(max_y_overlap, bottom_img.shape[0] - 100, 1300)  # Up to config max for full-size documents
    y_step = 20  # Reasonable steps for large images
    
    x_range = min(max_x_offset, min(top_img.shape[1], bottom_img.shape[1]) // 8)
    x_step = 10  # 10px steps for horizontal alignment
    
    log_func(f"    Searching overlap: Y={y_start}-{y_end}px (step {y_step}), X=±{x_range}px (step {x_step})")
    log_func(f"    Using confidence thresholds: primary={confidence_threshold}, backup={backup_threshold}")
    
    # Track best results for debugging
    top_candidates = []
    
    for y_overlap in range(y_start, y_end, y_step):
        # Get strips for comparison
        top_strip = top_gray[-y_overlap:, :]
        bottom_strip = bottom_gray[:y_overlap, :]
        
        if top_strip.shape[0] != bottom_strip.shape[0] or top_strip.shape[0] < 50:
            continue
            
        # Try different horizontal offsets
        for x_offset in range(-x_range, x_range + 1, x_step):
            try:
                # Calculate overlap region
                if x_offset >= 0:
                    # Bottom image shifts right
                    overlap_width = min(top_strip.shape[1] - x_offset, bottom_strip.shape[1])
                    if overlap_width < min(top_strip.shape[1], bottom_strip.shape[1]) * 0.6:  # Need 60% width overlap
                        continue
                    top_crop = top_strip[:, x_offset:x_offset + overlap_width]
                    bottom_crop = bottom_strip[:, :overlap_width]
                else:
                    # Bottom image shifts left
                    abs_offset = abs(x_offset)
                    overlap_width = min(top_strip.shape[1], bottom_strip.shape[1] - abs_offset)
                    if overlap_width < min(top_strip.shape[1], bottom_strip.shape[1]) * 0.6:
                        continue
                    top_crop = top_strip[:, :overlap_width]
                    bottom_crop = bottom_strip[:, abs_offset:abs_offset + overlap_width]
                
                # Use multiple scoring methods for text documents
                scores = []
                
                # Method 1: Direct pixel correlation (for identical text)
                correlation = cv2.matchTemplate(top_crop.astype(np.float32), 
                                              bottom_crop.astype(np.float32), 
                                              cv2.TM_CCOEFF_NORMED)[0, 0]
                scores.append(('correlation', correlation, 0.4))
                
                # Method 2: Edge matching (for text structure)
                top_edges = cv2.Canny(top_crop, 100, 200)
                bottom_edges = cv2.Canny(bottom_crop, 100, 200)
                
                edge_correlation = cv2.matchTemplate(top_edges.astype(np.float32),
                                                   bottom_edges.astype(np.float32),
                                                   cv2.TM_CCOEFF_NORMED)[0, 0]
                scores.append(('edges', edge_correlation, 0.3))
                
                # Method 3: Text line alignment (horizontal patterns)
                top_h_proj = np.sum(top_crop < 200, axis=1)  # Horizontal projection
                bottom_h_proj = np.sum(bottom_crop < 200, axis=1)
                
                if len(top_h_proj) == len(bottom_h_proj) and len(top_h_proj) > 0:
                    # Normalize projections
                    top_norm = top_h_proj / (np.max(top_h_proj) + 1)
                    bottom_norm = bottom_h_proj / (np.max(bottom_h_proj) + 1)
                    
                    # Calculate correlation
                    proj_corr = np.corrcoef(top_norm, bottom_norm)[0, 1]
                    if not np.isnan(proj_corr):
                        scores.append(('projection', proj_corr, 0.3))
                
                # Combine scores with weights
                if scores:
                    combined_score = sum(score * weight for _, score, weight in scores) / sum(weight for _, _, weight in scores)
                    
                    # Keep track of top candidates
                    top_candidates.append((combined_score, y_overlap, x_offset, scores))
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_overlap = y_overlap
                        best_offset = x_offset
                        
            except Exception as e:
                continue
    
    # Show top candidates for debugging
    top_candidates.sort(reverse=True)
    log_func(f"    Top overlap candidates:")
    for i, (score, y, x, score_details) in enumerate(top_candidates[:5]):  # Show top 5
        detail_str = ", ".join([f"{name}={val:.3f}" for name, val, _ in score_details])
        log_func(f"      {i+1}. Y={y}px, X={x}px, score={score:.3f} ({detail_str})")
    
    # Accept result if confidence is reasonable for document matching
    if best_score > confidence_threshold:
        log_func(f"    ✓ Found overlap: Y={best_overlap}px, X={best_offset}px (confidence: {best_score:.3f})")
        return best_overlap, best_offset, best_score
    else:
        log_func(f"    ✗ No reliable overlap found (best score: {best_score:.3f} < {confidence_threshold})")
        
        # If we're close to the backup threshold and have consistent Y values, accept it anyway
        if best_score > backup_threshold and len(top_candidates) >= 3:
            # Check if top candidates agree on Y position (within 40px)
            top_y_values = [y for _, y, _, _ in top_candidates[:3]]
            y_std = np.std(top_y_values)
            
            if y_std < 40:  # Candidates agree on position
                log_func(f"    ⚠ Low confidence but consistent position (Y std={y_std:.1f}), accepting result")
                return best_overlap, best_offset, best_score
        
        return None

def create_stitched_image_no_blend(top_img, bottom_img, y_overlap, x_offset, log_func):
    """
    Create the final stitched image with NO BLENDING - hard cut to preserve text quality.
    Place bottom image ON TOP of top image in overlap region.
    """
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
    
    log_func(f"    Canvas size: {canvas_width}x{canvas_height}")
    log_func(f"    Top image position: ({top_x_start}, 0)")
    
    # Place top image completely first
    stitched[:top_h, top_x_start:top_x_start + top_w] = top_img
    
    # Place bottom image ON TOP of the overlap region, then continue below
    bottom_y_start = top_h - y_overlap
    stitched[bottom_y_start:bottom_y_start + bottom_h, bottom_x_start:bottom_x_start + bottom_w] = bottom_img
    
    log_func(f"    ✓ Top image: complete ({top_h}px)")
    log_func(f"    ✓ Bottom image: placed ON TOP starting at Y={bottom_y_start}px")
    log_func(f"    ✓ Bottom image OVERWRITES overlap region (Y={bottom_y_start} to Y={top_h})")
    log_func(f"    ✓ Result: seamless transition at Y={top_h - y_overlap}px")
    
    log_func(f"    ✓ NO BLEND stitching complete: {stitched.shape[1]}x{stitched.shape[0]}")
    return stitched

def final_crop_to_document(img_array, x_offset, log_func):
    """
    Simple final crop that removes edge artifacts based on the x_offset used in stitching.
    This ensures both sides are cropped equally to remove any misalignment artifacts.
    """
    log_func(f"    Final edge cleanup crop...")
    
    h, w = img_array.shape[:2]
    log_func(f"    Stitched size: {w}x{h}")
    log_func(f"    Using x_offset from stitching: {x_offset}px")
    
    # Use the absolute value of x_offset as the crop amount for both sides
    crop_amount = abs(x_offset)
    
    # Ensure we don't crop more than 15% of the width as a safety measure
    max_crop = int(w * 0.15)
    if crop_amount > max_crop:
        log_func(f"    Limiting crop from {crop_amount}px to {max_crop}px (15% of width)")
        crop_amount = max_crop
    
    # Apply symmetric crop from both sides
    if crop_amount > 0 and crop_amount < w // 2:
        left_crop = crop_amount
        right_crop = w - crop_amount
        cropped = img_array[:, left_crop:right_crop]
        log_func(f"    ✓ Edge cleanup crop: {w}x{h} → {cropped.shape[1]}x{cropped.shape[0]} (removed {crop_amount}px from each side)")
        return cropped
    else:
        log_func(f"    ⚠ No crop needed or crop amount too large, keeping original")
        return img_array

def simple_concatenate(top_img, bottom_img, log_func):
    """Simple concatenation fallback when overlap detection fails."""
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
    log_func(f"    ✓ Simple concatenation: {stitched.shape[1]}x{stitched.shape[0]}")
    return stitched, 0  # Return 0 x_offset for simple concatenation

if __name__ == '__main__':
    process()
