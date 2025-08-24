import click
from pathlib import Path
import cv2
import numpy as np
import sys
import time

# Import shared logging utility
from .logging_utils import setup_script_logging, get_script_logger

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
SCRIPT_NAME = "deskew"

# Global logger - will be set up in main()
logger = None


def load_config(config_path=None):
    """Load configuration from TOML file."""
    if config_path is None:
        config_path = Path("config.toml")

    # Ensure it's a Path object
    if not isinstance(config_path, Path):
        config_path = Path(config_path)

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print("Error loading config: %s", e)
        print("Copy config.sample.toml to config.toml and edit as appropriate")
        sys.exit(1)


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--config", type=click.Path(path_type=Path), help="Path to config.toml file"
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output for debugging"
)
def main(input_dir, output_dir, config, verbose):
    """Deskew scanned document images."""

    # Load configuration
    try:
        config_data = load_config(config)
    except Exception as e:
        print("✗ Failed to load configuration: %s", e)
        return

    # Setup logging using shared utility
    global logger
    logger = setup_script_logging(SCRIPT_NAME, config_data, output_dir, verbose)

    logger.info("Starting deskew script...")
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Log target: %s", config_data['debug']['log_target'])
    logger.info(
        "Deskew config: angle_threshold=%s°, max_rotation=%s°",
        config_data["deskew"]["angle_threshold"],
        config_data["deskew"]["max_rotation"],
    )

    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("✓ Output directory created: %s", output_dir)
    except OSError:
        logger.error("✗ Failed to create output directory: %s", exc_info=True)
        return

    # Find image files (multiple formats supported)
    logger.info("Scanning for image files...")
    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".jp2",
    }
    image_files = []

    try:
        for file in input_dir.iterdir():
            if file.suffix.lower() in image_extensions:
                image_files.append(file)
                logger.debug("  Found: %s", file.name)
    except OSError:
        logger.error("Error scanning input directory", exc_info=True)
        return

    image_files.sort()  # Alphabetical ordering

    if not image_files:
        logger.error("No image files found in input directory")
        logger.error("Looking for files with extensions: %s", image_extensions)
        return

    # Group files by extension for summary
    extension_counts = {}
    for file in image_files:
        ext = file.suffix.lower()
        extension_counts[ext] = extension_counts.get(ext, 0) + 1

    summary = ", ".join(
        [f"{count} {ext}" for ext, count in sorted(extension_counts.items())]
    )
    logger.debug(
        "Found %s image files: %s",
        len(image_files),
        summary
    )

    # Process each image individually
    processed_count = 0
    failed_count = 0

    for i, image_file in enumerate(image_files, 1):
        logger.info(
            "Processing %s/%s: %s",
            i, len(image_files), image_file.name,
        )

        start_time = time.time()

        try:
            # Load image with OpenCV
            logger.debug("  Loading image: %s", image_file)
            img = cv2.imread(str(image_file))

            if img is None:
                logger.error("  Could not load image: %s", image_file)
                failed_count += 1
                continue

            logger.debug(
                "  ✓ Loaded image: %sx%s pixels",
                img.shape[1],
                img.shape[0],
            )

            # Deskew the image
            logger.debug("  Analyzing skew...")
            deskewed_img, angle, crop_pixels = deskew_image(img, config_data["deskew"])

            # Save final result as PNG (converts from any input format)
            logger.debug("  Saving final image...")
            final_path = output_dir / f"{image_file.stem}.png"

            # Use high quality PNG compression
            cv2.imwrite(str(final_path), deskewed_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            elapsed_time = time.time() - start_time
            input_format = image_file.suffix.upper()
            logger.info(
                "  ✓ Completed in %.2fs - %s→PNG - angle: %.3f°, crop: %spx - saved as %s",
                elapsed_time, input_format, angle, crop_pixels, final_path.name
            )
            processed_count += 1

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error("✗ Error processing %s after %.2fs",
                image_file.name, elapsed_time, exc_info=True
            )
            failed_count += 1
            continue

    # Summary
    logger.info("\n✨ Processing complete!")
    logger.info("Successfully processed: %s files", processed_count)
    if failed_count > 0:
        logger.error("Failed to process: %s files", failed_count)
    logger.info("Results saved to: %s", output_dir)


def deskew_image(img_array, processing_config):
    """
    Detect and correct document rotation using Hough line detection.
    Returns tuple of (processed_image, angle, crop_pixels)
    """
    angle_threshold = processing_config["angle_threshold"]
    max_rotation = processing_config["max_rotation"]

    logger.debug("    Analyzing image for skew...")

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    # Detect rotation angle with timeout
    try:
        detected_angle = detect_rotation_angle(gray)
    except Exception as e:
        logger.error("    Error in angle detection: %s", e)
        return img_array, 0.0, 0

    if detected_angle is None:
        logger.debug("    No significant rotation detected")
        return img_array, 0.0, 0

    logger.debug("    Detected angle: %.3f°", detected_angle)

    if abs(detected_angle) < angle_threshold:
        logger.debug(
            "    Angle %.3f° < threshold %s. Skipping rotation.",
            detected_angle, angle_threshold,
        )
        return img_array, detected_angle, 0

    if abs(detected_angle) > max_rotation:
        logger.debug(
            "    Angle %.3f° > max rotation %s°, skipping to avoid overcorrection.",
            detected_angle, max_rotation,
        )
        return img_array, detected_angle, 0

    # Rotate the image
    logger.debug("    Rotating image by %.3f°...", detected_angle)

    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)

    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, detected_angle, 1.0)

    # Crop to original dimensions and remove rotation wedges
    rotated = cv2.warpAffine(
        img_array,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # Calculate crop boundaries to remove rotation artifacts
    angle_rad = np.radians(abs(detected_angle))

    # Mathematical calculation for crop amount to remove rotation wedges
    # For a rotation, crop only top and bottom: crop_y = w * sin(angle) / 2
    if angle_rad < np.pi / 4:  # Less than 45 degrees
        crop_y = int(w * np.sin(angle_rad) / 2)  # Halved crop from top and bottom

        # Apply the mathematically calculated crop (top/bottom only)
        if crop_y > 0:
            # Ensure we don't crop beyond image boundaries
            crop_y = min(crop_y, h // 2)

            cropped = rotated[
                crop_y : h - crop_y, :
            ]  # Only crop top/bottom, keep full width
            logger.debug(
                "    ✓ Rotated by %.3f° and cropped %spx from top/bottom",
                detected_angle, crop_y
            )
            logger.debug(
                "    ✓ Final size: %sx%s → %sx%s",
                w, h, cropped.shape[1], cropped.shape[0],
            )
            return cropped, detected_angle, crop_y

    # Fallback: return uncropped rotation
    logger.debug("    ✓ Rotated by %.3f° (no crop applied)", detected_angle)
    return rotated, detected_angle, 0


def detect_rotation_angle(gray, timeout_seconds=10):
    """
    Detect document rotation angle using multiple Hough line detection methods.
    Added timeout to prevent hanging.
    """
    start_time = time.time()

    # Apply mild blur to reduce noise but preserve line structure
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use conservative edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    logger.debug(
        "    Image shape: %s, Edge pixels: %s",
        gray.shape, np.sum(edges > 0),
    )

    # Try multiple detection methods with timeout checking
    methods_tried = []

    # Method 1: Conservative probabilistic Hough
    if time.time() - start_time > timeout_seconds:
        logger.warning("    Timeout reached, using partial results")
        return None

    logger.debug("    Method 1: Conservative probabilistic Hough...")
    angle1 = try_probabilistic_hough_conservative(edges)
    if angle1 is not None:
        methods_tried.append(("Conservative Prob", angle1))

    # Method 2: Aggressive probabilistic Hough
    if time.time() - start_time > timeout_seconds:
        if methods_tried:
            logger.warning("    Timeout reached, using available results")
            return methods_tried[0][1]
        return None

    logger.debug("    Method 2: Aggressive probabilistic Hough...")
    angle2 = try_probabilistic_hough_aggressive(edges)
    if angle2 is not None:
        methods_tried.append(("Aggressive Prob", angle2))

    # Method 3: Standard Hough (skip if we already have good results)
    if len(methods_tried) < 2 and time.time() - start_time <= timeout_seconds:
        logger.debug("    Method 3: Standard Hough...")
        angle3 = try_standard_hough(edges)
        if angle3 is not None:
            methods_tried.append(("Standard", angle3))

    # Method 4: Projection method (only if no other methods worked)
    if not methods_tried and time.time() - start_time <= timeout_seconds:
        logger.debug("    Method 4: Projection profile (simplified)...")
        angle4 = try_projection_method_simple(gray)
        if angle4 is not None:
            methods_tried.append(("Projection", angle4))

    if not methods_tried:
        logger.error("    No method detected any angles!")
        return None

    # Display all results
    logger.debug("    Results from  methods: %s", len(methods_tried))
    for method, angle in methods_tried:
        logger.debug("      %s: %.3f°", method, angle)

    # Choose the best result
    if len(methods_tried) == 1:
        final_angle = methods_tried[0][1]
        logger.info("    Using only available result: %.3f°", final_angle)
    else:
        # Filter out any obviously wrong results (> 15°)
        reasonable_angles = [(m, a) for m, a in methods_tried if abs(a) <= 15]

        if not reasonable_angles:
            logger.info("    All detected angles > 15°, skipping")
            return None

        # If multiple reasonable results, take median
        angles = [a for _, a in reasonable_angles]
        final_angle = np.median(angles)
        logger.info(
            "    Using median of %s reasonable results: %.3f°",
            len(reasonable_angles), final_angle,
        )

    return final_angle


def try_probabilistic_hough_conservative(edges):
    """Conservative probabilistic Hough line detection."""
    try:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=150,
            maxLineGap=30,
        )
        return process_hough_lines(lines, "conservative")
    except Exception as e:
        logger.error("      Error in conservative Hough: %s", e)
        return None


def try_probabilistic_hough_aggressive(edges):
    """More aggressive probabilistic Hough to catch subtle lines."""
    try:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,  # Higher angular resolution
            threshold=15,  # Much lower threshold
            minLineLength=80,  # Shorter lines
            maxLineGap=50,  # Larger gaps
        )
        return process_hough_lines(lines, "aggressive")
    except Exception as e:
        logger.error("      Error in aggressive Hough: %s", e)
        return None


def process_hough_lines(lines, method_name):
    """Process Hough lines to extract rotation angle."""
    if lines is None:
        logger.debug("      No lines found with %s method", method_name)
        return None

    # Limit processing to first 100 lines to prevent hanging
    lines_to_process = lines[:100] if len(lines) > 100 else lines
    angles = []

    for x1, y1, x2, y2 in lines_to_process[:, 0]:
        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 5:
            continue

        angle = np.degrees(np.arctan2(dy, dx))

        # Normalize angle to [-90, 90]
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # Accept reasonable range
        if abs(angle) <= 20:
            angles.append(angle)

    logger.debug(
        "      Found %s valid lines with %s method (from %s total)",
        len(angles), method_name, len(lines_to_process),
    )

    if len(angles) >= 2:
        # Filter outliers if we have enough lines
        if len(angles) > 10:
            filtered_angles = remove_outliers(angles)
        else:
            filtered_angles = angles

        final_median = np.median(filtered_angles)
        logger.info(
            "      %s result: %.3f° from %s lines",
            method_name.capitalize(), final_median, len(filtered_angles),
        )

        return final_median

    logger.info("      Not enough valid lines for %s method", method_name)
    return None


def remove_outliers(angles):
    """Remove outliers using interquartile range method."""
    if len(angles) < 6:
        return angles

    # Use wider IQR multiplier to be less aggressive
    q1, q3 = np.percentile(angles, [20, 80])
    iqr = q3 - q1
    lower_bound = q1 - 2.0 * iqr
    upper_bound = q3 + 2.0 * iqr

    filtered = [a for a in angles if lower_bound <= a <= upper_bound]

    # Don't filter too aggressively - keep at least 50% of the data
    if len(filtered) < len(angles) * 0.5:
        return angles

    return filtered


def try_standard_hough(edges):
    """Standard Hough line detection with flexible parameters."""
    try:
        # Try with different thresholds
        for threshold in [40, 25, 15]:
            lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=threshold)

            if lines is not None and len(lines) >= 3:
                logger.debug(
                    "      Standard Hough found %s lines with threshold %s",
                    len(lines), threshold,
                )

                # Limit processing to prevent hanging
                lines_to_process = lines[:50] if len(lines) > 50 else lines
                angles = []

                for rho, theta in lines_to_process[:, 0]:
                    angle_degrees = np.degrees(theta) - 90
                    if abs(angle_degrees) <= 20:
                        angles.append(angle_degrees)

                if len(angles) >= 2:
                    filtered_angles = (
                        remove_outliers(angles) if len(angles) > 3 else angles
                    )
                    median_angle = np.median(filtered_angles)
                    logger.debug(
                        "      Standard result: %.3f° from %s lines",
                        median_angle, len(filtered_angles),
                    )
                    return median_angle

        logger.debug("      Standard Hough found no suitable lines")
        return None
    except Exception as e:
        logger.error("      Error in standard Hough: %s", e)
        return None


def try_projection_method_simple(gray):
    """Simplified projection profile method for angle detection."""
    try:
        h, w = gray.shape

        # Try smaller angle range with coarser steps to prevent hanging
        angles_to_try = np.arange(-5, 5.1, 0.5)
        best_angle = 0
        best_score = 0

        logger.debug("      Testing %s angles (simplified)...", len(angles_to_try))

        for angle in angles_to_try:
            # Rotate image (smaller image for speed)
            small_h, small_w = min(h, 500), min(w, 500)
            small_gray = cv2.resize(gray, (small_w, small_h))

            rotation_matrix = cv2.getRotationMatrix2D(
                (small_w // 2, small_h // 2), angle, 1.0
            )
            rotated = cv2.warpAffine(
                small_gray, rotation_matrix, (small_w, small_h), borderValue=255
            )

            # Calculate horizontal projection
            projection = np.sum(rotated < 200, axis=1)

            # Calculate variance (higher = better text line alignment)
            if len(projection) > 0:
                score = np.var(projection)
                if score > best_score:
                    best_score = score
                    best_angle = angle

        logger.debug(
            "      Best projection angle: %.3f° (score: %.0f)",
            best_angle, best_score,
        )

        if best_score > 100:  # Lower threshold for simplified method
            return best_angle
        else:
            logger.debug("      Projection score too low (%.0f < 100)", best_score)
            return None

    except Exception as e:
        logger.error("      Error in projection method: %s", e)
        return None


if __name__ == "__main__":
    main()
