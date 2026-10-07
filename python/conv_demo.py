#!/usr/bin/env python3
"""Handout Part 8: image filters with the CUDA convolution, called from Python.

1. Applies a set of filters (blur, Gaussian, sharpen, Sobel edge detection,
   Laplacian, emboss) to sample images, checks every GPU result against a
   NumPy reference, and saves the filtered images plus a gallery per image.
2. Times 3 image sizes x 3 filter sizes three ways: the plain C program
   (bin/conv_cpu, one core, and bin/conv_cpu_omp), the CUDA executable
   (bin/conv_gpu) and this Python program through the shared library.

Put your own photos in conv/images/ (any format PIL reads); otherwise sample
images from scikit-image or a generated test pattern are used.

    python3 python/conv_demo.py      -> results/conv/*.png, results/conv_perf.csv
"""
import csv
import glob
import os
import subprocess
import sys
import time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from lab6 import Lab6  # noqa: E402

OUT = os.path.join(ROOT, "results", "conv")
os.makedirs(OUT, exist_ok=True)

g5 = np.array([1, 4, 6, 4, 1], np.float32)
FILTERS = {
    "box3": np.full((3, 3), 1 / 9, np.float32),
    "gaussian5": np.outer(g5, g5) / 256.0,
    "sharpen3": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
    "laplacian3": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32),
    "emboss3": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32),
}


def ref_conv(img, f):
    """NumPy reference: same semantics as the C and CUDA code (correlation, clamp-to-edge)."""
    n = f.shape[0]
    r = n // 2
    p = np.pad(img.astype(np.float64), r, mode="edge")
    M = img.shape[0]
    out = np.zeros((M, M))
    for i in range(n):
        for j in range(n):
            out += p[i:i + M, j:j + M] * f[i, j]
    return out.astype(np.float32)


def square(arr, M):
    h, w = arr.shape
    s = min(h, w)
    arr = arr[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return np.asarray(Image.fromarray(arr).resize((M, M), Image.BILINEAR), np.uint8)


def load_images(M):
    imgs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "conv", "images", "*"))):
        try:
            imgs[os.path.splitext(os.path.basename(path))[0]] = square(np.asarray(Image.open(path).convert("L")), M)
        except OSError:
            pass
    if imgs:
        return imgs
    try:
        from skimage import data, color
        imgs["camera"] = square(data.camera(), M)
        imgs["coins"] = square(data.coins(), M)
        imgs["astronaut"] = square((color.rgb2gray(data.astronaut()) * 255).astype(np.uint8), M)
        return imgs
    except Exception:
        pass
    y, x = np.mgrid[0:M, 0:M] / M
    pattern = 127 + 60 * np.sin(40 * x) * np.cos(25 * y) + 60 * ((x - .5) ** 2 + (y - .5) ** 2 < .08)
    return {"pattern": np.clip(pattern, 0, 255).astype(np.uint8)}


def to_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def edges_u8(x):
    x = np.abs(x)
    return (255 * x / max(x.max(), 1e-6)).astype(np.uint8)


gpu = Lab6()

# ---- 1. correctness and filtered images -----------------------------------
print("== filters (GPU tiled kernel vs NumPy reference) ==")
for name, img in load_images(1024).items():
    Image.fromarray(img).save(os.path.join(OUT, f"{name}_original.png"))
    shown = {"original": img}
    for fname, f in FILTERS.items():
        out, kms = gpu.conv2d(img, f, tiled=True)
        err = float(np.max(np.abs(out - ref_conv(img, f))))
        print(f"{name:>10} {fname:>10}  kernel {kms:7.3f} ms   max |GPU - ref| = {err:.2e}")
        if fname.startswith("sobel") or fname == "laplacian3":
            shown[fname] = edges_u8(out)
        else:
            shown[fname] = to_u8(out)
        Image.fromarray(shown[fname]).save(os.path.join(OUT, f"{name}_{fname}.png"))
    gx, _ = gpu.conv2d(img, FILTERS["sobel_x"])
    gy, _ = gpu.conv2d(img, FILTERS["sobel_y"])
    shown["sobel_magnitude"] = edges_u8(np.hypot(gx, gy))
    Image.fromarray(shown["sobel_magnitude"]).save(os.path.join(OUT, f"{name}_sobel_magnitude.png"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = list(shown)
        fig, axes = plt.subplots(2, (len(keys) + 1) // 2, figsize=(3 * ((len(keys) + 1) // 2), 6.4))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, k in zip(axes.ravel(), keys):
            ax.imshow(shown[k], cmap="gray", vmin=0, vmax=255)
            ax.set_title(k, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"gallery_{name}.png"), dpi=110)
        plt.close(fig)
    except ImportError:
        pass

# ---- 2. performance: C program vs CUDA executable vs Python + library -----
print("\n== performance (box filter N x N on an M x M image) ==")
rows = []
bin_dir = os.path.join(ROOT, "bin")
for M in (512, 2048, 4096):
    img = next(iter(load_images(M).values()))
    for N in (3, 7, 15):
        f = np.full((N, N), 1 / (N * N), np.float32)
        for exe in ("conv_cpu", "conv_cpu_omp"):
            path = os.path.join(bin_dir, exe)
            if os.path.exists(path):
                line = subprocess.run([path, str(M), str(N), "1"], capture_output=True, text=True).stdout.strip()
                impl, _, _, ms = line.split(",")[:4]
                rows.append((f"c_{impl}", M, N, float(ms), float(ms)))
        for line in subprocess.run([os.path.join(bin_dir, "conv_gpu"), str(M), str(N), "10"],
                                   capture_output=True, text=True).stdout.split():
            impl, _, _, kms, _, e2e, _ = line.split(",")
            rows.append((f"cuda_exe_{impl[4:]}", M, N, float(e2e), float(kms)))
        gpu.conv2d(img, f)                                # warm-up for this size
        for tiled in (False, True):
            t0 = time.perf_counter()
            _, kms = gpu.conv2d(img, f, tiled=tiled)
            call = (time.perf_counter() - t0) * 1e3
            rows.append((f"python_lib_{'tiled' if tiled else 'naive'}", M, N, call, kms))

print(f"{'impl':>22} {'M':>5} {'N':>3} {'end-to-end ms':>14} {'kernel ms':>10}")
for r in rows:
    print(f"{r[0]:>22} {r[1]:>5} {r[2]:>3} {r[3]:>14.3f} {r[4]:>10.3f}")
with open(os.path.join(ROOT, "results", "conv_perf.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["impl", "M", "N", "end_to_end_ms", "kernel_ms"])
    w.writerows(rows)
print("-> results/conv/ and results/conv_perf.csv")
gpu.close()
