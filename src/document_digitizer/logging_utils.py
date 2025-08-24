"""
Shared logging utility for image processing scripts.
Provides consistent logging setup across multiple scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any


def setup_script_logging(
    script_name: str, config: Dict[str, Any], output_dir: Path, verbose: bool = False
) -> logging.Logger:
    """
    Set up logging for a script based on configuration.

    Args:
        script_name: Name of the script (used for log file naming)
        config: Configuration dictionary with debug settings
        output_dir: Directory where log files should be written
        verbose: Whether to enable verbose console output

    Returns:
        Configured logger instance
    """
    # Create logger for this script
    logger = logging.getLogger(script_name)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Get logging config
    log_target = config.get("debug", {}).get("log_target", "stdout")

    if log_target == "file":
        # File logging setup
        log_file = output_dir / f"{script_name}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Add console handler if verbose
        if verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter("%(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        logger.setLevel(logging.DEBUG)
        print(f"Detailed logging to: {log_file}")

    else:
        # Console logging only
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger


def get_script_logger(script_name: str) -> logging.Logger:
    """
    Get the logger instance for a script.

    Args:
        script_name: Name of the script

    Returns:
        Logger instance (must be set up first with setup_script_logging)
    """
    return logging.getLogger(script_name)


def setup_simple_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up simple console-only logging for scripts that don't need config files.

    Args:
        verbose: Whether to enable debug-level logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("simple")
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger


class LoggerMixin:
    """
    Mixin class that provides logging capabilities to any class.

    Usage:
        class MyProcessor(LoggerMixin):
            def __init__(self, script_name='myprocessor'):
                self.setup_class_logging(script_name)

            def process(self):
                self.logger.info("Processing started")
                self.logger.debug("Debug details...")
    """

    def setup_class_logging(self, script_name: str):
        """Set up logging for this class instance."""
        self.logger = logging.getLogger(script_name)

    def get_logger(self) -> logging.Logger:
        """Get the logger for this class."""
        if not hasattr(self, "logger"):
            raise RuntimeError("Logging not set up. Call setup_class_logging() first.")
        return self.logger
