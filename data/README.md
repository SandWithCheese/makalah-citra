# Seismic YOLO Dataset

This directory contains the YOLO-format dataset for subsurface hazard detection.

## Structure

```
data/
├─ raw/         # Raw unprocessed seismic images (gitignored)
│  └─ *.tif, *.tiff, *.png, *.jpg
├─ processed/   # Intermediate tiled outputs (optional, gitignored)
├─ train/
│  ├─ images/   # Training seismic image tiles
│  └─ labels/   # YOLO annotations (.txt files)
└─ val/
   ├─ images/   # Validation seismic image tiles
   └─ labels/   # YOLO annotations (.txt files)
```

### Folder Purposes

- **`raw/`**: Store your original, unprocessed seismic images here (TIFF, PNG, JPG format)
  - These are your source files before any preprocessing
  - **Gitignored** to avoid committing large binary files
  - Keep organized by survey/project if you have multiple sources

- **`processed/`**: Optional intermediate storage for tiled outputs from `seismic_tiler.py`
  - Contains subdirectories per image with tiles, manifests, and CSV metadata
  - **Gitignored** to save repository space
  - You'll manually select/move tiles from here to `train/` and `val/`

- **`train/`** and **`val/`**: Final YOLO-ready datasets
  - Contains only the annotated tiles used for training/validation
  - Images and corresponding label files
  - **Gitignored** but you may track a sample subset for testing

## Annotation Format

Each `.txt` label file contains one annotation per line:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to [0, 1] relative to image dimensions.

## Classes (as defined in `data.yaml`)

- **0**: `fault` - Fault/fracture zones
- **1**: `gas_chimney` - Gas chimneys/bright spots
- **2**: `void` - Subsurface voids or cavities
- **3**: `weak_layer` - Weak or unstable layers

## Dataset Preparation Workflow

1. **Place Raw Images**: Put your original seismic sections in `data/raw/`
   - Supported formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`
   - Keep files organized (e.g., by survey, date, or region)

2. **Tile Generation**: Use `preprocessing/seismic_tiler.py` to convert raw images into tiles
   ```bash
   python preprocessing/seismic_tiler.py \
     --input data/raw/ \
     --output data/processed/ \
     --tile 640 \
     --overlap 0.2
   ```

3. **Review Tiles**: Check the generated tiles in `data/processed/`
   - Inspect normalized images and tile quality
   - Review manifest files for tile metadata

4. **Annotation**: Label tiles using tools like [LabelImg](https://github.com/tzutalin/labelImg) or [CVAT](https://www.cvat.ai/)
   - Focus on tiles containing geological features
   - Export annotations in YOLO format (normalized bounding boxes)

5. **Split Dataset**: Manually distribute annotated tiles into train/val directories
   ```bash
   # Example: Move annotated tiles
   mv data/processed/section_01/*_x*_y*.png data/train/images/
   mv data/processed/section_01/*_x*_y*.txt data/train/labels/
   
   mv data/processed/section_02/*_x*_y*.png data/val/images/
   mv data/processed/section_02/*_x*_y*.txt data/val/labels/
   ```
   - Recommended split: 90% train, 10% validation
   - **Important**: Don't mix tiles from the same source image across splits

6. **Verification**: Check that each image has a corresponding label file
   ```bash
   # Count images and labels (should match)
   ls data/train/images/*.png | wc -l
   ls data/train/labels/*.txt | wc -l
   ```

## Important Notes

- Ensure train/val split doesn't mix tiles from the same seismic section (avoid data leakage)
- Blank or uninformative tiles should be filtered during preprocessing
- Percentile clipping normalizes intensity for consistent training
