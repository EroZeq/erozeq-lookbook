#!/usr/bin/env python3
"""
Automatic asset compressor for EroZeq's Assets Lookbook.
Generates optimized 'Lite' versions of images in `assets/lite/` mirroring `assets/images/`
and writes file size metadata to `assets/manifest.json`.
"""

import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image, GifImagePlugin, ImageFile
from PIL._binary import o8

ROOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ROOT_DIR / "assets" / "images"
LITE_DIR = ROOT_DIR / "assets" / "lite"
MANIFEST_PATH = ROOT_DIR / "assets" / "manifest.json"

ORIG_WRITE_LOCAL_HEADER = GifImagePlugin._write_local_header


def set_gif_lzw_bits(bits: int) -> None:
    """Configure Pillow's GIF frame writer to use `bits`-bit LZW minimum code size."""
    def custom_write_frame_data(fp, im_frame, offset, params):
        try:
            im_frame.encoderinfo = params
            ORIG_WRITE_LOCAL_HEADER(fp, im_frame, offset, 0)
            fp.seek(fp.tell() - 1)
            fp.write(o8(bits))
            ImageFile._save(
                im_frame,
                fp,
                [ImageFile._Tile("gif", (0, 0) + im_frame.size, 0, ("P", bits))],
            )
            fp.write(b"\0")
        finally:
            del im_frame.encoderinfo

    GifImagePlugin._write_frame_data = custom_write_frame_data


def format_size(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    if mb >= 0.1:
        return f"{mb:.2f} MB"
    kb = num_bytes / 1024
    return f"{kb:.1f} KB"


def optimize_gif_bytes(src_path: Path, max_colors: int = 64, denoise_thresh: int = 8) -> bytes:
    """Optimize animated or static GIF using global N-bit palette, temporal delta stabilization, and inter-frame transparency."""
    bits = 6 if max_colors <= 64 else (7 if max_colors <= 128 else 8)
    set_gif_lzw_bits(bits)

    im = Image.open(src_path)
    n_frames = getattr(im, "n_frames", 1)

    if n_frames <= 1:
        buf = io.BytesIO()
        set_gif_lzw_bits(8)
        im.save(buf, format="GIF", optimize=True)
        return buf.getvalue()

    rgba_frames = []
    durations = []
    canvas = Image.new("RGBA", im.size, (0, 0, 0, 0))

    for i in range(n_frames):
        im.seek(i)
        dur = im.info.get("duration", 50) or 50
        frame_rgba = im.convert("RGBA")
        disposal = getattr(im, "disposal_method", 1)
        if i == 0 or disposal == 2:
            canvas = frame_rgba.copy()
        else:
            canvas.alpha_composite(frame_rgba)
        rgba_frames.append(canvas.copy())
        durations.append(dur)

    avg_dur = sum(durations) / len(durations)
    if len(rgba_frames) >= 300 and avg_dur <= 50:
        step = 2
    elif len(rgba_frames) >= 100 and avg_dur <= 60:
        step = 2
    else:
        step = 1

    if step > 1:
        rgba_frames = [rgba_frames[i] for i in range(0, len(rgba_frames), step)]
        durations = [sum(durations[i : i + step]) for i in range(0, len(durations), step)]

    sample_indices = np.linspace(0, len(rgba_frames) - 1, min(8, len(rgba_frames)), dtype=int)
    strip = Image.new("RGB", (rgba_frames[0].width, rgba_frames[0].height * len(sample_indices)))
    for idx, fi in enumerate(sample_indices):
        strip.paste(rgba_frames[fi].convert("RGB"), (0, idx * rgba_frames[0].height))

    pal_img = strip.quantize(
        colors=max_colors - 1,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    )
    palette = pal_img.getpalette()[: (max_colors - 1) * 3] + [0, 0, 0]
    trans_idx = max_colors - 1
    pal_256 = palette + [0, 0, 0] * (256 - max_colors)
    pal_img.putpalette(pal_256)

    out_frames = []
    prev_rgb = None
    prev_q = None

    for f in rgba_frames:
        arr = np.array(f)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3] >= 128
        q = f.convert("RGB").quantize(palette=pal_img, dither=Image.Dither.NONE)
        q_arr = np.array(q)
        q_arr[~alpha] = trans_idx

        if prev_rgb is not None and denoise_thresh > 0:
            near_same = (
                (np.max(np.abs(rgb.astype(np.int16) - prev_rgb.astype(np.int16)), axis=-1) <= denoise_thresh)
                & alpha
                & (prev_q != trans_idx)
            )
            q_arr[near_same] = prev_q[near_same]

        prev_rgb = rgb.copy()
        prev_q = q_arr.copy()

        out_p = Image.fromarray(q_arr, mode="P")
        out_p.putpalette(palette)
        out_p.info["transparency"] = trans_idx
        out_frames.append(out_p)

    buf = io.BytesIO()
    out_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=out_frames[1:],
        optimize=True,
        palette=bytes(palette),
        duration=durations,
        loop=0,
        transparency=trans_idx,
        disposal=1,
    )
    return buf.getvalue()


def optimize_png_bytes(src_path: Path) -> bytes:
    """Optimize PNG using high-quality 256-color quantization + zlib level 9."""
    im = Image.open(src_path)
    candidates = []

    # Candidate 1: Pure lossless PNG optimization
    buf_lossless = io.BytesIO()
    im.save(buf_lossless, format="PNG", optimize=True, compress_level=9)
    candidates.append(buf_lossless.getvalue())

    # Candidate 2: Visually lossless 256-color quantized PNG (like pngquant)
    try:
        q = im.quantize(
            colors=256,
            method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.FLOYDSTEINBERG,
        )
        buf_quant = io.BytesIO()
        q.save(buf_quant, format="PNG", optimize=True, compress_level=9)
        candidates.append(buf_quant.getvalue())
    except Exception:
        pass

    return min(candidates, key=len)


def optimize_jpg_bytes(src_path: Path) -> bytes:
    """Optimize JPEG with progressive encoding and optimal Huffman tables."""
    im = Image.open(src_path)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
    return buf.getvalue()


def optimize_webp_bytes(src_path: Path) -> bytes:
    """Optimize static or animated WebP."""
    im = Image.open(src_path)
    n_frames = getattr(im, "n_frames", 1)
    if n_frames <= 1:
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=80, method=4)
        return buf.getvalue()

    frames = []
    durations = []
    for i in range(n_frames):
        im.seek(i)
        frames.append(im.copy())
        durations.append(im.info.get("duration", 50) or 50)

    avg_dur = sum(durations) / len(durations)
    if len(frames) >= 100 and avg_dur <= 60:
        step = 2
    else:
        step = 1

    if step > 1:
        frames = [frames[i] for i in range(0, len(frames), step)]
        durations = [sum(durations[i : i + step]) for i in range(0, len(durations), step)]

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        quality=75,
        method=4,
    )
    return buf.getvalue()


def process_all_assets(force: bool = False) -> Dict[str, Any]:
    LITE_DIR.mkdir(parents=True, exist_ok=True)

    existing_manifest: Dict[str, Any] = {}
    if MANIFEST_PATH.exists() and not force:
        try:
            existing_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("assets", {})
        except Exception:
            existing_manifest = {}

    manifest_assets: Dict[str, Any] = {}
    total_normal = 0
    total_lite = 0

    all_files = sorted(
        p for p in IMAGES_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")
    )

    for src_path in all_files:
        rel_from_images = src_path.relative_to(IMAGES_DIR)
        normal_rel = f"assets/images/{rel_from_images.as_posix()}"
        lite_rel = f"assets/lite/{rel_from_images.as_posix()}"
        dst_path = LITE_DIR / rel_from_images
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        orig_size = src_path.stat().st_size
        orig_mtime = int(src_path.stat().st_mtime)

        cached = existing_manifest.get(normal_rel)
        if (
            not force
            and cached
            and cached.get("normalSize") == orig_size
            and dst_path.exists()
        ):
            lite_size = dst_path.stat().st_size
            manifest_assets[normal_rel] = cached
            total_normal += orig_size
            total_lite += lite_size
            continue

        ext = src_path.suffix.lower()
        print(f"Compressing {normal_rel} ({format_size(orig_size)})...", flush=True)

        try:
            real_fmt = ""
            try:
                with Image.open(src_path) as probe_im:
                    real_fmt = (probe_im.format or "").upper()
            except Exception:
                real_fmt = ""

            if real_fmt == "GIF" or (not real_fmt and ext == ".gif"):
                opt_bytes = optimize_gif_bytes(src_path, max_colors=64, denoise_thresh=8)
            elif real_fmt == "PNG" or (not real_fmt and ext == ".png"):
                opt_bytes = optimize_png_bytes(src_path)
            elif real_fmt == "JPEG" or (not real_fmt and ext in (".jpg", ".jpeg")):
                opt_bytes = optimize_jpg_bytes(src_path)
            elif real_fmt == "WEBP" or (not real_fmt and ext == ".webp"):
                opt_bytes = optimize_webp_bytes(src_path)
            else:
                opt_bytes = None
        except Exception as exc:
            print(f"  [WARN] Failed to optimize {normal_rel}: {exc}", flush=True)
            opt_bytes = None

        if opt_bytes is not None and len(opt_bytes) < orig_size:
            dst_path.write_bytes(opt_bytes)
            lite_size = len(opt_bytes)
            effective_lite_rel = lite_rel
        else:
            if dst_path.exists():
                dst_path.unlink()
            lite_size = orig_size
            effective_lite_rel = normal_rel

        saved_pct = round((1 - (lite_size / orig_size)) * 100) if orig_size > 0 else 0
        print(
            f"  -> {format_size(lite_size)} (saved {saved_pct}%)",
            flush=True,
        )

        manifest_assets[normal_rel] = {
            "normalPath": normal_rel,
            "litePath": effective_lite_rel,
            "normalSize": orig_size,
            "liteSize": lite_size,
            "normalSizeFormatted": format_size(orig_size),
            "liteSizeFormatted": format_size(lite_size),
            "savedPercent": saved_pct,
            "mtime": orig_mtime,
        }
        total_normal += orig_size
        total_lite += lite_size

    total_saved_pct = round((1 - (total_lite / total_normal)) * 100) if total_normal > 0 else 0
    manifest_data = {
        "summary": {
            "totalNormalBytes": total_normal,
            "totalLiteBytes": total_lite,
            "totalNormalFormatted": format_size(total_normal),
            "totalLiteFormatted": format_size(total_lite),
            "totalSavedFormatted": format_size(max(0, total_normal - total_lite)),
            "totalSavedPercent": total_saved_pct,
        },
        "assets": manifest_assets,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(
        f"\nDone! Total Normal: {format_size(total_normal)} -> Total Lite: {format_size(total_lite)} "
        f"(Saved {format_size(max(0, total_normal - total_lite))}, -{total_saved_pct}%)"
    )
    return manifest_data


if __name__ == "__main__":
    force_run = "--force" in sys.argv
    process_all_assets(force=force_run)
