# Installation Guide

This guide covers installing the Document Digitizer toolkit and its dependencies.

## System Requirements

- **Python**: 3.9 or higher (as specified in pyproject.toml)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 2GB RAM minimum (4GB+ recommended for large images)
- **Storage**: Varies by document collection size

## Dependencies

The toolkit automatically installs these dependencies:

- **OpenCV** (`opencv-python>=4.8`): Image processing and computer vision
- **NumPy** (`numpy>=1.24`): Numerical computations
- **Click** (`click>=8.0`): Command-line interface
- **Rich** (`rich>=13.0`): Enhanced console output
- **Pillow** (`pillow>=10.0`): Additional image format support
- **tomli** (`tomli>=2.0`): TOML configuration parsing (Python < 3.11 only)

## Installation Methods

### Method 1: Standard Installation with Virtual Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/cliveholloway/document_digitizer.git
cd document_digitizer

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install the package
pip install -e .

# Verify installation
document-deskew --help
document-stitch --help
```

### Method 2: Development Installation

For contributors and developers:

```bash
# Clone and setup as above
git clone https://github.com/cliveholloway/document_digitizer.git
cd document_digitizer
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install with development dependencies
pip install -e ".[dev]"

# Format code
black src/

# Lint code
flake8 src/
```

## Command-Line Scripts

The package provides convenient command-line scripts:

```bash
document-deskew input_dir output_dir -v
document-deskew --help
```

## Platform-Specific Instructions

### Ubuntu/Debian Linux

```bash
# Install system dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Optional: Install OpenCV system dependencies for better performance
sudo apt install libopencv-dev

# Create virtual environment and install
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+
brew install python

# Verify Python version
python3 --version

# Create virtual environment and install
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Windows

1. **Install Python 3.9+** from [python.org](https://python.org)
   - ✅ Check "Add Python to PATH" during installation
   - ✅ Check "pip" is included

2. **Open Command Prompt or PowerShell**

3. **Create virtual environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install -e .
   ```

**Windows Notes:**
- Use Command Prompt or PowerShell, not Git Bash for virtual environments
- You may need Visual Studio Build Tools for OpenCV compilation
- Consider using Windows Subsystem for Linux (WSL2) for better compatibility

## Verifying Installation

After installation, verify everything works:

```bash
# Ensure virtual environment is activated
# You should see (venv) in your prompt

# Check installed scripts
document.deskew --help
document.stitch --help

# Test with sample data
cp config.sample.toml config.toml
python -m document_digitizer.deskew data/sample_scans test_output -v
```

Expected output should show:
- Configuration loaded successfully
- Image files found and processed
- Completion message with processing statistics

## Configuration Setup

1. **Copy the sample configuration:**
   ```bash
   cp config.sample.toml config.toml
   ```
   Edit as needed. Generally you will only want to tweak the `log_target`.

2. **Edit settings** (see [Configuration Guide](configuration.md)):
   ```toml
   [debug]
   log_target = "stdout"  # or "file"
   
   [deskew]
   angle_threshold = 0.1
   max_rotation = 10.0
   ```

3. **Test configuration:**
   ```bash
   document-deskew --help
   ```

## Virtual Environment Management

### Activating the Environment

**Every time** you want to use the tools:

```bash
# Linux/macOS
cd /path/to/document_digitizer
source venv/bin/activate

# Windows
cd C:\path\to\document_digitizer
venv\Scripts\activate
```

You should see `(venv)` in your command prompt when activated.

### Deactivating the Environment

```bash
deactivate
```

### Removing the Environment

```bash
# Simply delete the venv directory
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

## Troubleshooting Installation

### Common Issues

**Virtual environment not activating:**
```bash
# Ensure you're in the project directory
cd document_digitizer

# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
```

**"Command not found" after installation:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Check if scripts were installed
ls venv/bin/document-*  # Linux/macOS
dir venv\Scripts\document-*  # Windows

# Use module syntax as fallback
python -m document_digitizer.deskew --help
```

**OpenCV installation issues:**
```bash
# In your activated venv
pip uninstall opencv-python
pip install opencv-python-headless

# Or try specific version
pip install opencv-python==4.8.1.78
```

**Permission errors:**
```bash
# Ensure you're using virtual environment, not global install
source venv/bin/activate
pip install -e .

# Check that you're not using sudo (Linux/macOS)
```

**Python version issues:**
```bash
# Check Python version (must be 3.9+)
python --version

# Use specific Python version if needed
python3.9 -m venv venv
# or
python3.10 -m venv venv
```
