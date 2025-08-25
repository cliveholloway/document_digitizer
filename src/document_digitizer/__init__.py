"""
Document Processing CLI Tools

A collection of command-line tools for processing scanned document images:
- deskew.py: Correct rotation/skew in scanned images
- stitch.py: Combine paired images with overlap detection

Usage:
    document-deskew input_dir output_dir [--config config.toml] [--verbose]
    document-stitch input_dir output_dir [--config config.toml] [--verbose]

Or directly:
    python -m document_digitizer.deskew input_dir output_dir
    python -m document_digitizer.stitch input_dir output_dir
"""

__version__ = "0.1.0"
__author__ = "Clive Holloway"
__description__ = "Document image processing CLI tools for deskewing and stitching"
