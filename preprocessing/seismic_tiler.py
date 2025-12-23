import json, csv, argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image, ImageOps


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_image(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGB")
    return im


def percentile_clip_normalize(
    arr: np.ndarray, low_pct: float, high_pct: float
) -> np.ndarray:
    if arr.ndim == 2:
        lo, hi = np.percentile(arr, [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        return (arr * 255.0).round().astype(np.uint8)
    out = np.empty_like(arr, dtype=np.uint8)
    for c in range(arr.shape[2]):
        ch = arr[..., c]
        lo, hi = np.percentile(ch, [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1.0
        ch = np.clip(ch, lo, hi)
        ch = (ch - lo) / (hi - lo)
        out[..., c] = (ch * 255.0).round().astype(np.uint8)
    return out


def image_to_array(im: Image.Image) -> np.ndarray:
    if im.mode == "RGB":
        arr = np.asarray(im, dtype=np.float32)
    elif im.mode == "L":
        arr = np.asarray(im, dtype=np.float32)
    else:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    return arr


def stddev_tile(arr_uint8: np.ndarray) -> float:
    if arr_uint8.ndim == 3:
        return float(arr_uint8.reshape(-1, arr_uint8.shape[2]).std())
    return float(arr_uint8.std())


def iterate_tiles(W: int, H: int, tile: int, overlap: float):
    stride = max(1, int(tile * (1.0 - overlap)))
    y = 0
    while y < H:
        x = 0
        while x < W:
            x1 = min(x + tile, W)
            y1 = min(y + tile, H)
            x0 = max(0, x1 - tile)
            y0 = max(0, y1 - tile)
            yield x0, y0, x1 - x0, y1 - y0
            if x + stride >= W and x1 < W:
                x = W
            else:
                x += stride
        if y + stride >= H and y1 < H:
            y = H
        else:
            y += stride


def process_image(img_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    img_name = Path(img_path).stem
    out_dir = Path(cfg["output_dir"]) / img_name
    ensure_dir(out_dir)

    im = load_image(img_path)
    if cfg["to_grayscale"]:
        im = ImageOps.grayscale(im)

    arr = image_to_array(im)
    norm = percentile_clip_normalize(arr, cfg["clip_low_pct"], cfg["clip_high_pct"])
    norm_img = Image.fromarray(
        norm if norm.ndim == 2 else norm, mode=("L" if norm.ndim == 2 else "RGB")
    )

    full_out_path = None
    if cfg["save_normalized_full"]:
        full_out_path = str(out_dir / f"{img_name}_normalized.png")
        norm_img.save(full_out_path, optimize=True)

    H, W = norm.shape[0], norm.shape[1]
    tile = int(cfg["tile_size"])
    overlap = float(cfg["overlap"])

    tiles_meta: List[Dict[str, Any]] = []
    tile_counter = 0

    for x, y, w, h in iterate_tiles(W, H, tile, overlap):
        tile_img = norm_img.crop((x, y, x + w, y + h))
        arr_tile = np.asarray(tile_img)
        if cfg["skip_blank_tiles"]:
            if stddev_tile(arr_tile) < cfg["blank_stddev_thresh"]:
                continue
        if w != tile or h != tile:
            pad_img = Image.new(tile_img.mode, (tile, tile))
            pad_img.paste(tile_img, (0, 0))
            tile_img = pad_img

        tile_name = f"{img_name}_x{x}_y{y}_{tile}p.png"
        tile_path = str(out_dir / tile_name)
        tile_img.save(tile_path, optimize=True)
        tiles_meta.append(
            {
                "image": img_name,
                "tile_path": tile_path,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "tile_size": tile,
            }
        )
        tile_counter += 1

    meta = {
        "image": img_name,
        "input_path": img_path,
        "width": W,
        "height": H,
        "tile_size": tile,
        "overlap": overlap,
        "normalized_full_path": full_out_path,
        "tiles_count": tile_counter,
    }

    with open(out_dir / f"{img_name}_manifest.json", "w") as f:
        json.dump({"meta": meta, "tiles": tiles_meta}, f, indent=2)

    with open(out_dir / f"{img_name}_tiles.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "tile_path", "x", "y", "w", "h", "tile_size"]
        )
        writer.writeheader()
        for row in tiles_meta:
            writer.writerow(row)

    return meta


def run_batch(cfg: Dict[str, Any]):
    in_dir = Path(cfg["input_dir"])
    out_dir = Path(cfg["output_dir"])
    ensure_dir(out_dir)

    metas = []
    for p in sorted(in_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in cfg["accepted_exts"]:
            metas.append(process_image(str(p), cfg))
    with open(out_dir / "batch_manifest.json", "w") as f:
        json.dump({"config": cfg, "images": metas}, f, indent=2)
    return metas


def parse_args():
    """Parse command-line arguments for tiling configuration"""
    ap = argparse.ArgumentParser(
        description="Tile seismic images from data/raw to data/processed"
    )
    
    # Fixed input/output paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_input = project_root / "data" / "raw"
    default_output = project_root / "data" / "processed"
    
    ap.add_argument(
        "--tile", 
        type=int, 
        default=1024,
        help="Tile size in pixels (default: 1024)"
    )
    ap.add_argument(
        "--overlap", 
        type=float, 
        default=0.25,
        help="Overlap fraction between tiles (default: 0.25)"
    )
    ap.add_argument(
        "--clip-low", 
        type=float, 
        default=1.0,
        help="Lower percentile for intensity clipping (default: 1.0)"
    )
    ap.add_argument(
        "--clip-high", 
        type=float, 
        default=99.0,
        help="Upper percentile for intensity clipping (default: 99.0)"
    )
    ap.add_argument(
        "--grayscale", 
        type=int, 
        default=0,
        help="Convert to grayscale: 0=no, 1=yes (default: 0)"
    )
    ap.add_argument(
        "--skip-blank", 
        type=int, 
        default=1,
        help="Skip blank tiles: 0=no, 1=yes (default: 1)"
    )
    ap.add_argument(
        "--blank-std", 
        type=float, 
        default=3.0,
        help="Standard deviation threshold for blank detection (default: 3.0)"
    )
    ap.add_argument(
        "--save-full", 
        type=int, 
        default=1,
        help="Save normalized full image: 0=no, 1=yes (default: 1)"
    )
    
    args = ap.parse_args()
    
    # Build configuration dictionary with fixed paths
    cfg = {
        "input_dir": str(default_input),
        "output_dir": str(default_output),
        "tile_size": args.tile,
        "overlap": args.overlap,
        "clip_low_pct": args.clip_low,
        "clip_high_pct": args.clip_high,
        "to_grayscale": bool(args.grayscale),
        "skip_blank_tiles": bool(args.skip_blank),
        "blank_stddev_thresh": args.blank_std,
        "save_normalized_full": bool(args.save_full),
        "accepted_exts": [".tif", ".tiff", ".png", ".jpg", ".jpeg"],
    }
    
    return cfg


if __name__ == "__main__":
    print("="*70)
    print("SEISMIC IMAGE TILER")
    print("="*70)
    
    cfg = parse_args()
    
    print(f"\n📂 Input directory:  {cfg['input_dir']}")
    print(f"📂 Output directory: {cfg['output_dir']}")
    print(f"\n⚙️  Configuration:")
    print(f"   Tile size: {cfg['tile_size']}×{cfg['tile_size']} px")
    print(f"   Overlap: {cfg['overlap']*100:.0f}%")
    print(f"   Percentile clipping: {cfg['clip_low_pct']}-{cfg['clip_high_pct']}")
    print(f"   Grayscale conversion: {'Yes' if cfg['to_grayscale'] else 'No'}")
    print(f"   Skip blank tiles: {'Yes' if cfg['skip_blank_tiles'] else 'No'}")
    print(f"   Blank detection threshold: {cfg['blank_stddev_thresh']}")
    print("\n" + "="*70 + "\n")
    
    run_batch(cfg)
    
    print("\n" + "="*70)
    print("✅ TILING COMPLETE")
    print("="*70)
    print(f"\nProcessed tiles saved to: {cfg['output_dir']}")
    print("\nNext steps:")
    print("  1. Review tiles and manifests in data/processed/")
    print("  2. Annotate tiles using LabelImg or CVAT")
    print("  3. Export annotations in YOLO format")
    print("  4. Move annotated tiles to data/train/ and data/val/")

