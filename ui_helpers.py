import streamlit as st
import cv2
import numpy as np
import traceback


def load_image_from_upload(uploaded_file):
    """Đọc file upload của Streamlit thành ảnh OpenCV BGR."""
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Không thể decode ảnh. File có thể bị hỏng.")
            return None
        return img_bgr
    except Exception as e:
        st.error(f"Lỗi khi đọc ảnh: {e}")
        st.exception(traceback.format_exc())
        return None


def convert_image_for_streamlit(cv_img):
    """Chuyển ảnh OpenCV (BGR hoặc Xám) sang định dạng Streamlit (RGB hoặc Xám)."""
    if cv_img is None or cv_img.size == 0:
        return None

    if len(cv_img.shape) == 2:
        # Ảnh xám
        return cv_img
    if cv_img.shape[2] == 3:
        # Ảnh BGR, chuyển sang RGB
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    return cv_img  # Trường hợp khác