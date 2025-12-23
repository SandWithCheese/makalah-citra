# Raw Seismic Data

This folder contains **original, unprocessed seismic images** before any preprocessing.

## Purpose

Store your source seismic sections here in their original format:
- ✅ TIFF files (`.tif`, `.tiff`) - Preferred for high-quality data
- ✅ PNG files (`.png`) - Good for already-converted images
- ✅ JPG files (`.jpg`, `.jpeg`) - Acceptable but may have compression artifacts

## Organization Suggestions

### Option 1: By Survey/Project
```
raw/
├─ survey_2024_01/
│  ├─ line_001.tif
│  ├─ line_002.tif
│  └─ ...
└─ survey_2024_02/
   ├─ line_001.tif
   └─ ...
```

### Option 2: By Date
```
raw/
├─ 2024-01-15/
├─ 2024-02-20/
└─ ...
```

### Option 3: Flat Structure (simple)
```
raw/
├─ seismic_section_01.tif
├─ seismic_section_02.tif
└─ ...
```

## Important Notes

- ⚠️ **Gitignored**: Files in this folder are NOT tracked by git to avoid large binary commits
- 💾 **Keep backups**: Maintain original data elsewhere (external drive, cloud storage)
- 📏 **File size**: Typical seismic sections can be 50-500 MB each
- 🎨 **Grayscale vs RGB**: Both are supported, but grayscale is typical for seismic data

## Preprocessing Command

To convert raw images to YOLO-ready tiles:

```bash
python preprocessing/seismic_tiler.py \
  --input data/raw/ \
  --output data/processed/ \
  --tile 640 \
  --overlap 0.2 \
  --grayscale 1
```

See [`preprocessing/README.md`](../../preprocessing/README.md) for full documentation.

## File Naming Recommendations

Use descriptive, consistent names:
- ✅ `survey_offshore_2024_line_042.tif`
- ✅ `north_field_seismic_section_01.tif`
- ❌ `image.tif`
- ❌ `seismic (copy) final_v2.tif`

Good naming helps with:
- Traceability from tiles back to source
- Organization and searching
- Collaboration with team members
