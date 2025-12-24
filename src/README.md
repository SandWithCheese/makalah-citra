# YOLO Training Script

This directory contains the training script for YOLO object detection models for **local execution**.

---

## File

### `train_yolo.py`
**Local training script** - Streamlined Python script for training YOLO models on your local GPU.

**Features:**
- Automatic GPU detection
- Data extraction and validation
- Train/val split generation
- YOLO model training
- Comprehensive logging

---

## Quick Start

### 🖥️ Local Execution

**Recommended (handles CUDA libraries automatically):**
```bash
# Navigate to src directory
cd /path/to/makalah-citra/src

# Ensure you have your annotated dataset
ls ../data.zip  # Should contain images/, labels/, classes.txt

# Run training using the helper script
./run_train.sh \
  --data_zip ../data.zip \
  --model yolo11s.pt \
  --epochs 60 \
  --imgsz 640 \
  --batch 16
```

**Alternative (manual execution):**
```bash
# Set CUDA library path first
export LD_LIBRARY_PATH="$(pwd)/venv/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"

# Run training
python train_yolo.py \
  --data_zip ../data.zip \
  --model yolo11s.pt \
  --epochs 60 \
  --imgsz 640 \
  --batch 16
```

---

## Script Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_zip` | str | `data.zip` | Path to your zipped dataset |
| `--base_path` | str | current dir | Working directory for training |
| `--model` | str | `yolo11s.pt` | YOLO model variant (`yolo11s`, `yolo12s`, `yolo12m`) |
| `--epochs` | int | `60` | Number of training epochs |
| `--imgsz` | int | `640` | Training image size (640, 480, 1024, etc.) |
| `--batch` | int | `16` | Batch size (reduce if GPU memory insufficient) |
| `--device` | str | auto-detect | Device to use (`0` for GPU, `cpu` for CPU) |
| `--skip_extract` | flag | False | Skip extraction if data already extracted |

---

## Dataset Requirements

Your `data.zip` must contain:

```
data.zip
├── images/          # Image files (.png, .jpg, .tif)
├── labels/          # YOLO annotation files (.txt)
└── classes.txt      # Class names (one per line)
```

**Example `classes.txt`:**
```
fault
gas_chimney
void
weak_layer
```

**YOLO label format** (`.txt` files):
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1].

---

## What the Script Does

1. **GPU Check** 🔍
   - Verifies CUDA availability
   - Reports GPU specs

2. **Data Extraction** 📦
   - Unzips `data.zip`
   - Validates dataset structure

3. **Train/Val Split** 📊
   - Randomly splits data (default: 90% train, 10% val)
   - Copies files to proper YOLO structure

4.  **Config Generation** ⚙️
    -   Creates `data.yaml` from `classes.txt`
    -   Sets correct paths for Ultralytics

5.  **Training** 🚀
    -   Installs Ultralytics if needed
    -   Trains YOLO model with specified parameters
    -   Saves results to `runs/detect/train/`

---

## Output Structure

After training:

```
makalah-citra/
├── src/
│   └── train_yolo.py
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── data.yaml                    # Generated config
├── custom_data/                 # Extracted dataset (temp)
└── runs/
    └── detect/
        └── train/
            ├── weights/
            │   ├── best.pt      # ⭐ Best model
            │   └── last.pt      # Last epoch
            ├── results.png      # Training curves
            ├── confusion_matrix.png
            └── ... (other metrics)
```

---

## Examples

### Example 1: Quick Local Training

```bash
./run_train.sh --data_zip ../data.zip --epochs 40
```

### Example 2: High-Resolution Training

```bash
./run_train.sh \
  --data_zip ../data.zip \
  --model yolo11m.pt \
  --imgsz 1024 \
  --epochs 80 \
  --batch 8
```

### Example 3: CPU Training (No GPU)

```bash
./run_train.sh \
  --data_zip ../data.zip \
  --device cpu \
  --batch 4 \
  --epochs 20
```

### Example 4: Re-training After Data Changes

```bash
# First run extracts data
./run_train.sh --data_zip ../new_data.zip --epochs 50

# If you modify labels manually in custom_data/, skip re-extraction
./run_train.sh --skip_extract --epochs 50
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size
```bash
./run_train.sh --batch 8
# or even
./run_train.sh --batch 4
```

### Issue: Script says "data.zip not found"

**Solution:** Provide full path
```bash
./run_train.sh --data_zip /absolute/path/to/data.zip
```

### Issue: Poor validation performance

**Solutions:**
- Increase dataset size (aim for 200+ images)
- Train longer (`--epochs 100`)
- Use larger model (`--model yolo12m.pt`)
- Check annotation quality

### Issue: CUDA Library Error (`libnvrtc-builtins.so.13.0` not found)

**Error message:**
```
nvrtc: error: failed to open libnvrtc-builtins.so.13.0.
  Make sure that libnvrtc-builtins.so.13.0 is installed correctly.
```

**Solution:** Always use the helper script `run_train.sh`, which sets the correct `LD_LIBRARY_PATH`.

---

## Advanced: Integration with Preprocessing

Complete workflow from raw seismic images:

```bash
# Step 1: Tile raw seismic images
# Place images in data/raw/ first
python preprocessing/seismic_tiler.py --tile 640 --overlap 0.2

# Step 2: Annotate tiles
# Use Label Studio (see main README)

# Step 3: Zip annotated data
# Create data.zip containing images/, labels/, and classes.txt

# Step 4: Train
cd src
./run_train.sh --data_zip ../data.zip --epochs 60
```

---

## Model Selection Guide

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolo12n.pt` | ~6 MB | Fastest | Lowest | Edge devices, real-time |
| `yolo12s.pt` | ~20 MB | Fast | Good | **Recommended starting point** |
| `yolo12m.pt` | ~45 MB | Medium | Better | More accuracy needed |
| `yolo12l.pt` | ~90 MB | Slow | Best | Maximum accuracy |

For seismic anomaly detection, **`yolo11s.pt` is recommended** as a good balance.

---

## Complete Training Workflow

### Prerequisites

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Verify GPU (recommended):**
    ```bash
    nvidia-smi
    ```

### Workflow Steps

1.  **Prepare raw data:**
    ```bash
    # Place seismic images in data/raw/
    ls ../data/raw/*.tif
    ```

2.  **Tile images:**
    ```bash
    python ../preprocessing/seismic_tiler.py --tile 640 --overlap 0.2
    ```

3.  **Annotate tiles:**
    -   Use Label Studio
    -   Export in YOLO format

4.  **Create dataset:**
    ```bash
    # Create data.zip with images/, labels/, classes.txt
    ```

5.  **Train model:**
    ```bash
    ./run_train.sh --data_zip ../data.zip --epochs 60
    ```

6.  **Evaluate:**
    ```bash
    yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml
    ```

7.  **Predict:**
    ```bash
    yolo detect predict \
      model=runs/detect/train/weights/best.pt \
      source=../data/processed/new_tiles/ \
      save=True
    ```

---

## Citation

Based on Ultralytics YOLO:
```
@software{yolo_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
}
```

---

## See Also

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Preprocessing README](../preprocessing/README.md)
- [Project Main README](../README.md)
- [Dataset Structure](../data/README.md)
