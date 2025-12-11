import cv2
import numpy as np
try:
    from numpy.lib.stride_tricks import sliding_window_view
except Exception as e:
    raise ImportError("Cần numpy>=1.20 để dùng sliding_window_view.") from e

from .utils import pad_image # Import hàm pad từ file utils cùng cấp

# ----------------- 1) Convert to gray -----------------
def rgb2gray(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
    return gray

# ----------------- 2) Convolution helper (vectorized) -----------------
def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    src = pad_image(img, ph, pw) # Dùng hàm pad_image đã import
    patches = sliding_window_view(src, (kh, kw))
    out = np.tensordot(patches, kernel, axes=([2, 3], [0, 1]))
    return out.astype(np.float32)

# ----------------- 3) Gaussian Blur (separable, vectorized) -----------------
def gaussian_kernel(k=5, sigma=1.0):
    if k % 2 == 0:
        raise ValueError("k must be odd")
    half = k // 2
    ax = np.arange(-half, half + 1, dtype=np.float32)
    kern1d = np.exp(-0.5 * (ax / float(sigma)) ** 2)
    kern1d = kern1d / kern1d.sum()
    return kern1d.astype(np.float32)

def gaussian_blur(img, k=5, sigma=1.0):
    kx = gaussian_kernel(k, sigma)
    pad = k // 2
    src = np.pad(img, ((0, 0), (pad, pad)), mode='reflect')
    patches_rows = sliding_window_view(src, k, axis=1)
    temp = np.tensordot(patches_rows, kx, axes=([2], [0]))
    src2 = np.pad(temp, ((pad, pad), (0, 0)), mode='reflect')
    patches_cols = sliding_window_view(src2, k, axis=0)
    out = np.tensordot(patches_cols, kx, axes=([2], [0]))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# ----------------- 4) Sobel (manual) -----------------
def sobel_filters(img):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve2d(img, kx)
    gy = convolve2d(img, ky)
    mag = np.hypot(gx, gy)
    if mag.max() == 0:
        mag_norm = mag.astype(np.uint8)
    else:
        mag_norm = (mag / mag.max() * 255).astype(np.uint8)
    return gx, gy, mag_norm

# ----------------- 5) Otsu threshold (manual) -----------------
def otsu_threshold(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    total = img.size
    sum_total = np.dot(np.arange(256), hist)
    weight_bg = 0.0
    sum_bg = 0.0
    max_var = 0.0
    thresh = 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0: continue
        weight_fg = total - weight_bg
        if weight_fg == 0: break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t
    return (img >= thresh).astype(np.uint8) * 255, thresh

# ----------------- 6) Morphology: vectorized erosion & dilation -----------------
def dilation(bin_img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    src = np.pad(bin_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    patches = sliding_window_view(src, (kh, kw))
    h, w = bin_img.shape
    patches_resh = patches.reshape(h, w, kh * kw)
    mask = kernel.flatten().astype(bool)
    out = np.any(patches_resh[..., mask] == 255, axis=2).astype(np.uint8) * 255
    return out

def erosion(bin_img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    src = np.pad(bin_img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    patches = sliding_window_view(src, (kh, kw))
    h, w = bin_img.shape
    patches_resh = patches.reshape(h, w, kh * kw)
    mask = kernel.flatten().astype(bool)
    out = np.all(patches_resh[..., mask] == 255, axis=2).astype(np.uint8) * 255
    return out

def closing(bin_img, kernel):
    return erosion(dilation(bin_img, kernel), kernel)

def opening(bin_img, kernel):
    return dilation(erosion(bin_img, kernel), kernel)

# ----------------- 7) Connected Components labeling (two-pass) -----------------
def connected_components(bin_img):
    h, w = bin_img.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 1
    parent = {}
    for i in range(h):
        for j in range(w):
            if bin_img[i, j] == 0: continue
            neighbors = []
            if i > 0 and labels[i - 1, j] > 0: neighbors.append(labels[i - 1, j])
            if j > 0 and labels[i, j - 1] > 0: neighbors.append(labels[i, j - 1])
            if not neighbors:
                labels[i, j] = label
                parent[label] = label
                label += 1
            else:
                m = min(neighbors)
                labels[i, j] = m
                for n in neighbors:
                    if n != m:
                        pa = parent.get(m, m)
                        pb = parent.get(n, n)
                        newp = min(pa, pb)
                        parent[pa] = newp
                        parent[pb] = newp
    for k in list(parent.keys()):
        r = k
        while parent[r] != r: r = parent[r]
        parent[k] = r
    new_id = {}
    cur = 1
    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                root = parent[labels[i, j]]
                if root not in new_id:
                    new_id[root] = cur
                    cur += 1
                labels[i, j] = new_id[root]
    return labels, cur - 1

# ----------------- 9) Grayscale Morphology (dùng cho Blackhat) -----------------
def gray_dilate(imgf, kx, ky):
    padx, pady = kx // 2, ky // 2
    src = np.pad(imgf, ((pady, pady), (padx, padx)), mode='reflect')
    patches = sliding_window_view(src, (ky, kx))
    out = patches.max(axis=(2, 3))
    return out.astype(imgf.dtype)

def gray_erode(imgf, kx, ky):
    padx, pady = kx // 2, ky // 2
    src = np.pad(imgf, ((pady, pady), (padx, padx)), mode='reflect')
    patches = sliding_window_view(src, (ky, kx))
    out = patches.min(axis=(2, 3))
    return out.astype(imgf.dtype)