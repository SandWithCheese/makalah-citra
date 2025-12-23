import os
import sys
import yaml
import argparse
import shutil
import zipfile
from pathlib import Path
import random

DEFAULT_BASE_PATH = os.getcwd()

print(f"📂 Working directory: {DEFAULT_BASE_PATH}")


def check_gpu():
    """Check GPU availability"""
    print("\n" + "="*60)
    print("GPU CHECK")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️  No CUDA GPU detected. Training will use CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed. Please install: pip install torch torchvision")
        return False


def extract_data(zip_path, output_dir):
    """Extract data.zip to custom_data folder"""
    print("\n" + "="*60)
    print("EXTRACTING DATASET")
    print("="*60)
    
    if not os.path.exists(zip_path):
        print(f"❌ Error: data.zip not found at {zip_path}")
        print(f"   Please provide the path to your dataset zip file.")
        sys.exit(1)
    
    print(f"📦 Extracting {zip_path}...")
    custom_data_dir = os.path.join(output_dir, 'custom_data')
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(custom_data_dir)
    
    print(f"✅ Extracted to {custom_data_dir}")
    
    # Verify structure
    required_items = ['images', 'labels', 'classes.txt']
    for item in required_items:
        item_path = os.path.join(custom_data_dir, item)
        if not os.path.exists(item_path):
            print(f"⚠️  Warning: {item} not found in extracted data")
    
    return custom_data_dir


def train_val_split(custom_data_dir, output_dir, train_pct=0.9):
    """Split data into train and validation sets"""
    print("\n" + "="*60)
    print("CREATING TRAIN/VAL SPLIT")
    print("="*60)
    
    # Use project's data directory (relative to script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_root = project_root / "data"
    
    # Create directory structure in project's data folder
    train_img_dir = data_root / "train" / "images"
    train_lbl_dir = data_root / "train" / "labels"
    val_img_dir = data_root / "val" / "images"
    val_lbl_dir = data_root / "val" / "labels"
    
    for directory in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    # Define paths to input dataset
    input_image_path = os.path.join(custom_data_dir, 'images')
    input_label_path = os.path.join(custom_data_dir, 'labels')
    
    # Get list of all images and annotation files (recursive search)
    img_file_list = [path for path in Path(input_image_path).rglob('*') 
                     if path.is_file() and path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    txt_file_list = [path for path in Path(input_label_path).rglob('*.txt') 
                     if path.is_file()]
    
    print(f"\n📊 Dataset information:")
    print(f"   Number of image files: {len(img_file_list)}")
    print(f"   Number of annotation files: {len(txt_file_list)}")
    
    # Determine number of files to move to each folder
    file_num = len(img_file_list)
    train_num = int(file_num * train_pct)
    val_num = file_num - train_num
    
    print(f"\n� Split distribution:")
    print(f"   Images moving to train: {train_num} ({train_pct*100:.0f}%)")
    print(f"   Images moving to validation: {val_num} ({(1-train_pct)*100:.0f}%)")
    
    # Select files randomly and copy them to train or val folders
    for i, set_num in enumerate([train_num, val_num]):
        for ii in range(set_num):
            # Randomly select an image from the list
            img_path = random.choice(img_file_list)
            img_fn = img_path.name
            base_fn = img_path.stem
            txt_fn = base_fn + '.txt'
            txt_path = os.path.join(input_label_path, txt_fn)
            
            # Determine destination folder
            if i == 0:  # Copy first set of files to train folders
                new_img_path, new_txt_path = train_img_dir, train_lbl_dir
            elif i == 1:  # Copy second set of files to validation folders
                new_img_path, new_txt_path = val_img_dir, val_lbl_dir
            
            # Copy image file
            shutil.copy(str(img_path), str(new_img_path / img_fn))
            
            # Copy label file if it exists (background images may not have labels)
            if os.path.exists(txt_path):
                shutil.copy(txt_path, str(new_txt_path / txt_fn))
            
            # Remove from list to avoid selecting it again
            img_file_list.remove(img_path)
    
    print("✅ Data split complete")
    return str(data_root)


def create_data_yaml(custom_data_dir, data_dir, output_dir):
    """Create YOLO data.yaml configuration file"""
    print("\n" + "="*60)
    print("CREATING DATA.YAML CONFIG")
    print("="*60)
    
    classes_file = os.path.join(custom_data_dir, 'classes.txt')
    
    if not os.path.exists(classes_file):
        print(f"❌ Error: classes.txt not found at {classes_file}")
        print("   Please ensure your dataset includes a classes.txt labelmap file")
        sys.exit(1)
    
    # Read class names
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    
    # Create data dictionary
    data = {
        'path': data_dir,
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(classes),
        'names': classes
    }
    
    # Write YAML
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"✅ Created config at: {yaml_path}")
    print(f"\n📋 Configuration:")
    print(f"   Classes ({len(classes)}): {classes}")
    print(f"   Train path: {data['train']}")
    print(f"   Val path: {data['val']}")
    
    return yaml_path


def install_ultralytics():
    """Install Ultralytics if not already installed"""
    try:
        import ultralytics
        print(f"✅ Ultralytics already installed (version {ultralytics.__version__})")
        return True
    except ImportError:
        print("📦 Installing Ultralytics...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
        print("✅ Ultralytics installed")
        return True


def train_model(data_yaml, model='yolo11s.pt', epochs=60, imgsz=640, batch=16, device=None):
    """Train YOLO model"""
    print("\n" + "="*60)
    print("TRAINING YOLO MODEL")
    print("="*60)
    
    # Install ultralytics if needed
    install_ultralytics()
    
    from ultralytics import YOLO
    
    # Auto-detect device if not specified
    if device is None:
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n🔧 Training configuration:")
    print(f"   Model: {model}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Device: {device}")
    print(f"   Config: {data_yaml}")
    
    # Load model
    model = YOLO(model)
    
    # Train
    print(f"\n🚀 Starting training...\n")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device
    )
    
    print("\n✅ Training complete!")
    print(f"   Results saved to: runs/detect/train")
    print(f"   Best model: runs/detect/train/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO model locally for seismic hazard detection'
    )
    parser.add_argument(
        '--data_zip',
        type=str,
        default='data.zip',
        help='Path to data.zip file (default: data.zip in current dir)'
    )
    parser.add_argument(
        '--base_path',
        type=str,
        default=DEFAULT_BASE_PATH,
        help=f'Base working directory (default: current directory)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo12s.pt',
        help='YOLO model to train (default: yolo12s.pt)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=60,
        help='Number of training epochs (default: 60)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Training image size (default: 640)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Training device (default: auto-detect)'
    )
    parser.add_argument(
        '--skip_extract',
        action='store_true',
        help='Skip data extraction (use existing custom_data)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SEISMIC YOLO TRAINING PIPELINE")
    print("="*60)
    print(f"Data source: {args.data_zip}")
    print(f"Working directory: {args.base_path}")
    
    # Check GPU
    check_gpu()
    
    # Extract data
    if not args.skip_extract:
        custom_data_dir = extract_data(args.data_zip, args.base_path)
    else:
        custom_data_dir = os.path.join(args.base_path, 'custom_data')
        print(f"\n⏭️  Skipping extraction, using: {custom_data_dir}")
    
    # Create train/val split
    data_dir = train_val_split(custom_data_dir, args.base_path, train_pct=0.9)
    
    # Create data.yaml
    data_yaml = create_data_yaml(custom_data_dir, data_dir, args.base_path)
    
    # Clean up temporary extraction directory
    if os.path.exists(custom_data_dir):
        print(f"\n🧹 Cleaning up temporary directory: {custom_data_dir}")
        shutil.rmtree(custom_data_dir)
        print("✅ Cleanup complete")
    
    # Train model
    train_model(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review training metrics in runs/detect/train/")
    print("  2. Test model with: yolo detect predict model=runs/detect/train/weights/best.pt source=<image>")
    print("  3. Validate model with: yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml")


if __name__ == '__main__':
    main()
