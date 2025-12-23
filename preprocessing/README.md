# Seismic Image Preprocessing

This directory contains the batch preprocessing pipeline for converting large seismic images into YOLO-compatible training tiles.

## Seismic Tiler (`seismic_tiler.py`)

**Batch processor** that converts entire directories of seismic sections into normalized, tiled datasets with complete metadata tracking.

---

## Features

### 🎯 Core Capabilities

1.  **Batch Processing**
    -   Processes entire directories of seismic images automatically
    -   Supports multiple formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`
    -   Creates organized output subdirectories per input image

2.  **Percentile-Based Normalization**
    -   Adaptive intensity clipping per image (default: 1st-99th percentile)
    -   Works with both grayscale and RGB images
    -   Channel-independent normalization for color images
    -   Robust to outliers in seismic amplitude data

3.  **Intelligent Tiling**
    -   Configurable tile size (default: 1024×1024)
    -   Overlap support to preserve spatial context (default: 25%)
    -   Smart padding for edge tiles to maintain consistent dimensions
    -   Efficient stride calculation to minimize redundancy

4.  **Blank Tile Filtering**
    -   Standard deviation-based quality check (default threshold: 3.0)
    -   Automatically skips uninformative regions (e.g., water column)
    -   Reduce dataset size and training waste

5.  **Metadata Generation**
    -   **Per-image manifests** (`{image_name}_manifest.json`) - Complete tile metadata
    -   **Tile CSV** (`{image_name}_tiles.csv`) - Tabular tile coordinates
    -   **Batch manifest** (`batch_manifest.json`) - Full processing configuration and summary
    -   Enables reproducibility and tile-to-source traceability

---

## Usage

### Basic Batch Processing

The tiler automatically reads from `data/raw/` and outputs to `data/processed/`.
**You must place your raw images in `data/raw/` first.**

```bash
# Run with default settings
# Input: data/raw/
# Output: data/processed/
python seismic_tiler.py
```

### Full Configuration Example

```bash
python seismic_tiler.py \
  --tile 1024 \
  --overlap 0.25 \
  --clip-low 1.0 \
  --clip-high 99.0 \
  --grayscale 1 \
  --skip-blank 1 \
  --blank-std 3.0 \
  --save-full 1
```

> **Note:** Input and output directories are fixed:
>
> -   **Input:** `data/raw/` (your raw seismic images)
> -   **Output:** `data/processed/` (generated tiles)

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tile` | int | `1024` | Tile size in pixels (square) |
| `--overlap` | float | `0.25` | Overlap fraction (0.0-0.5) between adjacent tiles |
| `--clip-low` | float | `1.0` | Lower percentile for intensity clipping |
| `--clip-high` | float | `99.0` | Upper percentile for intensity clipping |
| `--grayscale` | int | `0` | Convert to grayscale (0=no, 1=yes) |
| `--skip-blank` | int | `1` | Skip low-variance tiles (0=no, 1=yes) |
| `--blank-std` | float | `3.0` | Standard deviation threshold for blank detection |
| `--save-full` | int | `1` | Save normalized full-resolution image (0=no, 1=yes) |

---

## Output Structure

For each input image (e.g., `seismic_section_01.png`), the script creates:

```
data/processed/
└── seismic_section_01/
    ├── seismic_section_01_normalized.png     # Full normalized image
    ├── seismic_section_01_x0_y0_1024p.png   # Tile at (0,0)
    ├── seismic_section_01_x768_y0_1024p.png # Tile at (768,0) - 25% overlap
    ├── ...                                   # More tiles
    ├── seismic_section_01_manifest.json     # Complete metadata
    └── seismic_section_01_tiles.csv         # Tile coordinate table
```

### Manifest Structure (`_manifest.json`)

```json
{
  "meta": {
    "image": "seismic_section_01",
    "input_path": "data/raw/seismic_section_01.png",
    "width": 4096,
    "height": 2048,
    "tile_size": 1024,
    "overlap": 0.25,
    "normalized_full_path": "data/processed/seismic_section_01/seismic_section_01_normalized.png",
    "tiles_count": 42
  },
  "tiles": [
    {
      "image": "seismic_section_01",
      "tile_path": "data/processed/seismic_section_01/seismic_section_01_x0_y0_1024p.png",
      "x": 0,
      "y": 0,
      "w": 1024,
      "h": 1024,
      "tile_size": 1024
    }
  ]
}
```

---

## Why This Preprocessing Pipeline?

### 1. **Percentile Clipping (1-99%)**

**Problem**: Seismic amplitude data contains extreme outliers (e.g., multiples, noise bursts) that compress the dynamic range of geological features.

**Solution**: Percentile-based clipping removes outliers while preserving the majority distribution. Unlike fixed thresholds, this adapts to each image's statistics.

**Implementation**: Per-channel normalization supports both grayscale seismic sections and RGB-enhanced interpretations.

---

### 2. **Overlap Tiling (25% default)**

**Problem**: Subsurface anomalies (faults, gas chimneys) may cross tile boundaries, causing:
- Truncated features in training data
- Model inability to detect boundary-spanning objects
- Reduced spatial context

**Solution**: 25% overlap ensures most features appear complete in at least one tile while balancing dataset size.

**Trade-off**: Higher overlap = more context but larger dataset and slower training.

---

### 3. **Blank Tile Filtering (std < 3.0)**

**Problem**: Seismic images contain large uniform regions:
- Water column (constant velocity)
- Deep sedimentary basins (minimal features)
- Masked areas

**Solution**: Standard deviation filtering identifies low-information tiles automatically.

**Why std instead of variance?** Threshold is interpretable in pixel intensity units (0-255 scale).

---

### 4. **Metadata Tracking**

**Problem**: YOLO training loses connection between tiles and source images, preventing:
- Post-processing reassembly
- Error analysis at source scale
- Annotation validation

**Solution**: JSON manifests and CSV tables enable:
- Tile-to-source traceability
- Reproducible preprocessing
- Reconstruction of full-image predictions

---

## Integration with YOLO Workflow

### Step 1: Run Preprocessing

```bash
# Place your raw seismic images in data/raw/
# Then run the tiler
python seismic_tiler.py \
  --tile 640 \
  --overlap 0.2 \
  --grayscale 1
```

### Step 2: Annotate

Use **Label Studio** (see main README for setup):
1. Import images from `data/processed/<image_folder>/`
2. Label anomalies
3. Export in **YOLO format**

### Step 3: Organize for YOLO

Create a `data.zip` file containing:
- `images/`: The image tiles
- `labels/`: The YOLO txt files
- `classes.txt`: The class names

### Step 4: Train

```bash
# From src/ directory
./run_train.sh --data_zip ../data.zip --epochs 100 --imgsz 640
```

---

## Key Implementation Details

### Tile Iteration Algorithm

The `iterate_tiles()` function ensures:
- **Complete coverage** of the source image
- **Edge handling** via smart repositioning (tiles snap to image boundaries)
- **Consistent tile size** through padding when necessary

### Standard Deviation Calculation

```python
def stddev_tile(arr_uint8: np.ndarray) -> float:
    if arr_uint8.ndim == 3:  # RGB
        return float(arr_uint8.reshape(-1, arr_uint8.shape[2]).std())
    return float(arr_uint8.std())  # Grayscale
```

Computes global standard deviation across all channels to assess tile information content.

---

## Troubleshooting

### Issue: All tiles rejected (blank filter too aggressive)

**Solution**: Lower `--blank-std` threshold:
```bash
--blank-std 1.0  # More permissive
```

### Issue: Training images look washed out

**Solution**: Adjust percentile range:
```bash
--clip-low 2.0 --clip-high 98.0  # Preserve more dynamic range
```

### Issue: Tiles too small/large for GPU

**Solution**: Adjust tile size:
```bash
--tile 512   # Smaller for limited VRAM
--tile 1280  # Larger for high-res features
```

---

## Scientific Rationale

This preprocessing pipeline balances:
1.  **Geophysical domain knowledge** (seismic data characteristics)
2.  **Computer vision best practices** (normalization, context preservation)
3.  **Practical ML constraints** (dataset size, computational efficiency)

The approach treats seismic sections as **2D intensity fields** where:
- Vertical axis ≠ horizontal axis (depth vs. lateral distance)
- High-frequency noise is common and informative
- Amplitude outliers are artifacts, not geological signal

**Important**: This is preprocessing for pattern detection, not seismic processing (e.g., no migration, deconvolution, or filtering).

---

## Next Steps

1.  ✅ Run batch tiling
2.  📋 Review generated manifests
3.  🏷️ Annotate tiles with Label Studio
4.  📦 Package into data.zip
5.  🚀 Train with `src/run_train.sh`
