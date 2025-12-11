import cv2
import numpy as np


# ----------------- Utility: hình ảnh -----------------
def imread_color(path):
    """Đọc ảnh màu từ đường dẫn."""
    img = cv2.imread(path)
    if img is None:
        print(f"Lỗi: Không đọc được ảnh {path}")
        return None
    return img


def imshow(title, img):
    """Hiển thị ảnh (dùng cho debug)."""
    if img is None:
        print("None image")
        return
    h, w = img.shape[:2]
    max_h = 800
    if h > max_h:
        r = max_h / h
        img = cv2.resize(img, (int(w * r), max_h))
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_image(img, angle_deg, center=None):
    """Xoay ảnh (affine)."""
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    return rotated


def pad_image(img, pad_h, pad_w, mode='reflect'):
    """Hàm pad ảnh (dùng cho convolution)."""
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)