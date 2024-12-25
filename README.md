# ImageFingerprint-Modifier (IFMod)

A sophisticated image processing application that applies subtle modifications to images while preserving their visual appearance. The app is designed to process images through a pipeline of transformations that modify the image's digital fingerprint while maintaining visual fidelity.

## Features

- **User-Friendly Interface**: Drag and drop interface for easy image processing
- **Multi-Step Processing Pipeline**:
  - Metadata Modification: Adds and modifies image metadata
  - Geometric Transformations: Applies subtle geometric changes
  - Noise Addition: Introduces minimal noise patterns
  - Compression Adjustments: Modifies compression parameters
  - Alpha Channel Variations: Adds subtle alpha channel modifications

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- BMP
- TIFF
- WebP

## How to Use

1. Launch the application
2. Drag and drop an image or click to select one
3. Configure the processing pipeline using the checkboxes
4. Click "Modify image" to process
5. Save the modified image when ready

## Technical Details

- The application preserves visual quality while modifying the image's digital signature
- Each processed image receives a unique set of modifications
- Processed images maintain high visual fidelity to the original
- Hash verification ensures successful modification

## Requirements

- Python 3.x
- PIL (Python Imaging Library)
- NumPy
- TkinterDnD2

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

## License

© 2024 ImageFingerprint-Modifier. All rights reserved. 