import cv2
import numpy as np

# Giả định 'alg' được import từ file algorithms.py của bro
from . import algorithms as alg


# ----------------- Pipeline chính (Batch) -----------------
def detect_plate_manual(img_bgr, resize_w=1000):
    """
    Pipeline gốc của bro, dùng box thẳng.
    (HÀM NÀY GIỮ NGUYÊN NHƯ FILE BRO CUNG CẤP)
    """
    # --- BƯỚC 0: RESIZE ---
    h0, w0 = img_bgr.shape[:2]
    scale = resize_w / float(w0)
    img = cv2.resize(img_bgr, (resize_w, int(h0 * scale)), interpolation=cv2.INTER_AREA)

    # --- BƯỚC 1: GRAYSCALE ---
    gray = alg.rgb2gray(img)

    # --- BƯỚC 2: BLUR ---
    blurred = alg.gaussian_blur(gray, k=5, sigma=2.0)

    # --- BƯỚC 3: BLACKHAT ---
    gh = alg.gray_dilate(blurred.astype(np.float32), 15, 5)
    gh = alg.gray_erode(gh, 15, 5)
    blackhat = cv2.subtract(gh.astype(np.uint8), blurred.astype(np.uint8))

    # --- BƯỚC 4: SOBEL ---
    gx, gy, mag = alg.sobel_filters(blurred)

    # --- BƯỚC 5: KẾT HỢP ---
    combined = np.maximum(blackhat, mag)

    # --- BƯỚC 6: NHỊ PHÂN HÓA ---
    bin_img, thr = alg.otsu_threshold(combined)

    # --- BƯỚC 7: OPENING (LỌC NHIỄU) ---
    kernel_open = np.ones((5, 5), dtype=np.uint8)
    opened_img = alg.opening(bin_img, kernel_open)

    # --- BƯỚC 8: CLOSING (NỐI KÝ TỰ) ---
    kx, ky = 11, 5
    kernel_close = np.ones((ky, kx), dtype=np.uint8)
    closed_img = alg.closing(opened_img, kernel_close)

    # --- BƯỚC 9: TÌM BLOB & LỌC (LOGIC MỚI CỦA BRO) ---
    labels, num_labels = alg.connected_components(closed_img)

    candidates = []
    out_vis = img.copy()

    for i in range(1, num_labels + 1):
        rows, cols = np.where(labels == i)
        area = len(rows)
        if area < 1000: continue

        x, y = np.min(cols), np.min(rows)
        w_straight, h_straight = np.max(cols) - x + 1, np.max(rows) - y + 1
        ar = max(w_straight, h_straight) / max(min(w_straight, h_straight), 1e-6)
        extent = area / (w_straight * h_straight)

        # Ngưỡng lọc
        if area < 5000 or area > 50000: continue
        if not ((3.0 <= ar <= 5.5) or (1.0 <= ar <= 2.5)): continue
        if extent < 0.2: continue

        cv2.rectangle(out_vis, (x, y), (x + w_straight, y + h_straight), (0, 255, 0), 2)
        cv2.putText(out_vis, f"AR:{ar:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        candidates.append({
            'area': area,
            'bbox_straight': (x, y, w_straight, h_straight),
            'extent': extent
        })

    # --- BƯỚC 10: CHỌN & CẮT ẢNH ---
    best, plate_crop = None, None
    if candidates:
        best = max(candidates, key=lambda x: x['extent'])
        (x, y, w_straight, h_straight) = best['bbox_straight']
        cv2.rectangle(out_vis, (x, y), (x + w_straight, y + h_straight), (0, 0, 255), 2)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1] - 1, x + w_straight), min(img.shape[0] - 1, y + h_straight)
        plate_crop = img[y1:y2, x1:x2].copy()

    return out_vis, plate_crop


# ----------------- Pipeline cho Streamlit (Đã cập nhật) -----------------
def detect_plate_manual_steps(
    img_bgr,
    resize_w=1000,
    gaussian_k=5,
    gaussian_sigma=2.0,
    blackhat_kx=15,
    blackhat_ky=5,
    open_k=5,
    close_kx=11,
    close_ky=5,
    min_area=5000,
    max_area=50000,
    ar_range_1=(3.0, 5.5),
    ar_range_2=(1.0, 2.5),
    min_extent=0.2
):
    """
    Pipeline cho Streamlit, nhận tham số và dùng logic box thẳng.
    """
    steps = {}

    # --- BƯỚC 0: RESIZE ---
    h0, w0 = img_bgr.shape[:2]
    scale = resize_w / float(w0)
    img = cv2.resize(img_bgr, (resize_w, int(h0 * scale)), interpolation=cv2.INTER_AREA)
    steps['0_Resized'] = img

    # --- BƯỚC 1: GRAYSCALE ---
    gray = alg.rgb2gray(img)
    steps['1_Gray'] = gray

    # --- BƯỚC 2: BLUR ---
    blurred = alg.gaussian_blur(gray, k=gaussian_k, sigma=gaussian_sigma)
    steps['2_Blurred'] = blurred

    # --- BƯỚC 3: BLACKHAT ---
    blurred_f = blurred.astype(np.float32)
    gh = alg.gray_dilate(blurred_f, blackhat_kx, blackhat_ky)
    gh = alg.gray_erode(gh, blackhat_kx, blackhat_ky)
    blackhat = cv2.subtract(gh.astype(np.uint8), blurred.astype(np.uint8))
    steps['3_Blackhat'] = blackhat

    # --- BƯỚC 4: SOBEL ---
    gx, gy, mag = alg.sobel_filters(blurred)
    steps['4_Sobel_Mag'] = mag

    # --- BƯỚC 5: KẾT HỢP ---
    combined = np.maximum(blackhat, mag)
    steps['5_Combined'] = combined

    # --- BƯỚC 6: NHỊ PHÂN HÓA ---
    bin_img, thr = alg.otsu_threshold(combined)
    steps['6_Binary'] = bin_img

    # --- BƯỚC 7: OPENING (LỌC NHIỄU) ---
    kernel_open = np.ones((open_k, open_k), dtype=np.uint8)
    opened_img = alg.opening(bin_img, kernel_open)
    steps['7_Opened'] = opened_img

    # --- BƯỚC 8: CLOSING (NỐI KÝ TỰ) ---
    kernel_close = np.ones((close_ky, close_kx), dtype=np.uint8)
    closed_img = alg.closing(opened_img, kernel_close)
    steps['8_Closed'] = closed_img

    # --- BƯỚC 9: TÌM BLOB & LỌC (LOGIC MỚI CỦA BRO) ---
    labels, num_labels = alg.connected_components(closed_img)

    candidates = []
    out_vis = img.copy()

    for i in range(1, num_labels + 1):
        rows, cols = np.where(labels == i)
        area = len(rows)
        if area < 100: continue

        x, y = np.min(cols), np.min(rows)
        w_straight, h_straight = np.max(cols) - x + 1, np.max(rows) - y + 1
        ar = max(w_straight, h_straight) / max(min(w_straight, h_straight), 1e-6)
        extent = area / max((w_straight * h_straight), 1)

        # Áp dụng các ngưỡng lọc từ sidebar
        if area < min_area or area > max_area: continue
        if not ((ar_range_1[0] <= ar <= ar_range_1[1]) or (ar_range_2[0] <= ar <= ar_range_2[1])): continue
        if extent < min_extent: continue

        cv2.rectangle(out_vis, (x, y), (x + w_straight, y + h_straight), (0, 255, 0), 2)
        candidates.append({
            'area': area,
            'bbox_straight': (x, y, w_straight, h_straight),
            'extent': extent
        })
    steps['9_Contours'] = out_vis

    # --- BƯỚC 10: CHỌN & CẮT ẢNH ---
    best, plate_crop = None, None
    if candidates:
        best = max(candidates, key=lambda x: x['extent'])
        (x, y, w_straight, h_straight) = best['bbox_straight']
        cv2.rectangle(out_vis, (x, y), (x + w_straight, y + h_straight), (0, 0, 255), 2)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1] - 1, x + w_straight), min(img.shape[0] - 1, y + h_straight)
        plate_crop = img[y1:y2, x1:x2].copy()

    steps['10_Crop'] = plate_crop
    return steps